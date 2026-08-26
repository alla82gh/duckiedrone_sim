#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_preprocess.py

Offline preprocessing for DD21 PEM system identification.

Model to be identified:

    dx[k+1] = A_PEM dx[k] + B_PEM du[k]

State convention:

    x = [
        x, y, z,
        phi, theta, psi,
        vx, vy, vz,
        p, q, r
    ]

Actual physical plant input:

    u_phys = [
        T,
        tau_phi,
        tau_theta,
        tau_psi
    ]

Nominal MPC equilibrium input:

    u_e = [
        T_hover,
        0,
        0,
        0
    ]

where:

    T_hover = 6.22935 N

Preprocessing strategy
----------------------

1. Read raw asynchronous ROS streams.
2. Use excitation phase 1 as the stable pre-hover trim window.
3. Use excitation phase 2 as the identification window.
4. Construct a uniform 100 Hz time base.
5. Interpolate states linearly.
6. Apply zero-order hold (ZOH) to actual plant inputs.
7. Compute per-run empirical hover trim.
8. Center states and inputs around the per-run trim.
9. Build:

       X_k
       U_k
       X_{k+1}

10. Construct training and held-out validation datasets.

IMPORTANT:

Transitions are formed inside each run BEFORE concatenation.
Therefore no artificial transition is ever created between runs.

Author: Abdallah GHOUL
2026
"""

import argparse
import csv
import json
import os

import numpy as np


# ======================================================================
# Fixed DD21 identification convention
# ======================================================================

STATE_COLUMNS = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]

INPUT_COLUMNS = [
    "T",
    "tau_phi",
    "tau_theta",
    "tau_psi",
]

ANGLE_INDICES = [3, 4, 5]

NX = 12
NU = 4

TS_DEFAULT = 0.01

T_HOVER = 6.22935

U_E = np.array([
    T_HOVER,
    0.0,
    0.0,
    0.0,
], dtype=float)


# ======================================================================
# Frozen datasets
# ======================================================================

TRAIN_RUNS = [
    "pem_train_thrust_01",
    "pem_train_roll_01",
    "pem_train_pitch_01",
    "pem_train_yaw_01",
    "pem_train_mimo_01",
]

VALIDATION_RUNS = [
    "pem_val_mimo_01",
]


# ======================================================================
# CSV utilities
# ======================================================================

def read_numeric_csv(path, columns):
    """
    Read selected numeric columns from a CSV file.

    Returns
    -------
    dict
        column_name -> numpy array
    """

    with open(path, newline="") as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(
                "CSV has no header: {}".format(path)
            )

        missing = [
            c for c in columns
            if c not in reader.fieldnames
        ]

        if missing:
            raise RuntimeError(
                "Missing columns in {}: {}".format(
                    path,
                    missing
                )
            )

        data = {
            c: []
            for c in columns
        }

        for row in reader:

            for c in columns:
                data[c].append(
                    float(row[c])
                )

    return {
        c: np.asarray(v, dtype=float)
        for c, v in data.items()
    }


def matrix_from_dict(data, columns):
    """
    Stack selected dictionary columns into a matrix.
    """

    return np.column_stack([
        data[c]
        for c in columns
    ])


# ======================================================================
# Timestamp sanitation
# ======================================================================

def sort_keep_last(t, y):
    """
    Sort samples by timestamp and remove duplicate timestamps.

    If duplicate timestamps exist, the LAST sample is retained.

    Parameters
    ----------
    t : ndarray, shape (N,)
    y : ndarray, shape (N, m)

    Returns
    -------
    t_clean
    y_clean
    duplicate_count
    """

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) != len(y):
        raise RuntimeError(
            "Timestamp/data length mismatch."
        )

    order = np.argsort(
        t,
        kind="stable"
    )

    t = t[order]
    y = y[order]

    if len(t) == 0:
        raise RuntimeError(
            "Empty timestamp vector."
        )

    # Keep the final sample for each repeated timestamp.
    keep = np.ones(
        len(t),
        dtype=bool
    )

    if len(t) > 1:
        keep[:-1] = (
            t[:-1] != t[1:]
        )

    duplicate_count = int(
        len(t) - np.sum(keep)
    )

    t = t[keep]
    y = y[keep]

    if len(t) > 1:

        if not np.all(
            np.diff(t) > 0.0
        ):
            raise RuntimeError(
                "Timestamps are not strictly increasing "
                "after duplicate removal."
            )

    return (
        t,
        y,
        duplicate_count
    )


# ======================================================================
# Excitation phase utilities
# ======================================================================

def phase_window(exc_t, phase_code, wanted_phase):
    """
    Return first and last ROS timestamps for a phase.
    """

    mask = (
        np.rint(phase_code).astype(int)
        == int(wanted_phase)
    )

    if not np.any(mask):
        raise RuntimeError(
            "Excitation phase {} not found.".format(
                wanted_phase
            )
        )

    tw = exc_t[mask]

    return (
        float(np.min(tw)),
        float(np.max(tw))
    )


# ======================================================================
# Uniform time base
# ======================================================================

def build_uniform_grid(t0, t1, ts):
    """
    Construct a globally aligned uniform time grid.

    The grid is aligned to multiples of Ts.
    """

    if t1 <= t0:
        raise RuntimeError(
            "Invalid synchronization window: "
            "t1 <= t0."
        )

    # Align to global Ts grid.
    start = (
        np.ceil(
            (t0 - 1.0e-10) / ts
        )
        * ts
    )

    end = (
        np.floor(
            (t1 + 1.0e-10) / ts
        )
        * ts
    )

    if end <= start:
        raise RuntimeError(
            "Synchronization window too short."
        )

    n = (
        int(
            np.floor(
                (end - start) / ts
                + 1.0e-9
            )
        )
        + 1
    )

    t = (
        start
        + np.arange(
            n,
            dtype=float
        )
        * ts
    )

    return t


# ======================================================================
# State interpolation
# ======================================================================

def interpolate_states(
        state_t,
        state_x,
        target_t):
    """
    Linear interpolation of the 12-state vector.

    Euler angles are unwrapped before interpolation.
    """

    x_work = np.array(
        state_x,
        dtype=float,
        copy=True
    )

    # Robust handling if angles ever cross +/- pi.
    for j in ANGLE_INDICES:

        x_work[:, j] = np.unwrap(
            x_work[:, j]
        )

    x_target = np.empty(
        (
            len(target_t),
            NX
        ),
        dtype=float
    )

    for j in range(NX):

        x_target[:, j] = np.interp(
            target_t,
            state_t,
            x_work[:, j]
        )

    return x_target


# ======================================================================
# Input zero-order hold
# ======================================================================

def zoh_inputs(
        input_t,
        input_u,
        target_t):
    """
    Zero-order hold interpolation for actual plant inputs.
    """

    idx = (
        np.searchsorted(
            input_t,
            target_t,
            side="right"
        )
        - 1
    )

    if np.any(idx < 0):
        raise RuntimeError(
            "ZOH target precedes first input sample."
        )

    if np.any(
        idx >= len(input_t)
    ):
        raise RuntimeError(
            "Invalid ZOH index."
        )

    return input_u[idx, :]


# ======================================================================
# Synchronization
# ======================================================================

def synchronize_window(
        state_t,
        state_x,
        input_t,
        input_u,
        requested_t0,
        requested_t1,
        ts):
    """
    Synchronize state and input streams over a requested window.

    State:
        linear interpolation

    Input:
        zero-order hold
    """

    t0 = max(
        float(requested_t0),
        float(state_t[0]),
        float(input_t[0])
    )

    t1 = min(
        float(requested_t1),
        float(state_t[-1]),
        float(input_t[-1])
    )

    if t1 <= t0:
        raise RuntimeError(
            "No common state/input synchronization window."
        )

    t = build_uniform_grid(
        t0,
        t1,
        ts
    )

    x = interpolate_states(
        state_t,
        state_x,
        t
    )

    u = zoh_inputs(
        input_t,
        input_u,
        t
    )

    return (
        t,
        x,
        u
    )


# ======================================================================
# Validation
# ======================================================================

def validate_finite(name, a):
    """
    Assert that an array contains finite values only.
    """

    if not np.all(
        np.isfinite(a)
    ):
        raise RuntimeError(
            "{} contains NaN or Inf.".format(
                name
            )
        )


def validate_uniform_time(t, ts):
    """
    Validate the synchronized sampling interval.
    """

    if len(t) < 2:
        raise RuntimeError(
            "Too few synchronized samples."
        )

    dt = np.diff(t)

    if not np.allclose(
        dt,
        ts,
        atol=1.0e-9,
        rtol=0.0
    ):
        raise RuntimeError(
            "Non-uniform synchronized time base."
        )


# ======================================================================
# Run processing
# ======================================================================

def process_run(
        run_name,
        role,
        data_root,
        output_root,
        ts):
    """
    Preprocess one identification run.
    """

    run_dir = os.path.join(
        data_root,
        run_name
    )

    if not os.path.isdir(run_dir):
        raise RuntimeError(
            "Run directory not found: {}".format(
                run_dir
            )
        )

    state_path = os.path.join(
        run_dir,
        "state_raw.csv"
    )

    input_path = os.path.join(
        run_dir,
        "input_raw.csv"
    )

    exc_path = os.path.join(
        run_dir,
        "excitation_raw.csv"
    )

    for path in [
        state_path,
        input_path,
        exc_path,
    ]:

        if not os.path.isfile(path):
            raise RuntimeError(
                "Missing file: {}".format(
                    path
                )
            )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    state_data = read_numeric_csv(
        state_path,
        ["t_ros"] + STATE_COLUMNS
    )

    state_t_raw = state_data["t_ros"]

    state_x_raw = matrix_from_dict(
        state_data,
        STATE_COLUMNS
    )

    state_t, state_x, state_duplicates = (
        sort_keep_last(
            state_t_raw,
            state_x_raw
        )
    )

    # ------------------------------------------------------------------
    # Actual plant input
    # ------------------------------------------------------------------

    input_data = read_numeric_csv(
        input_path,
        [
            "t_ros",
            "T",
            "tau_phi",
            "tau_theta",
            "tau_psi",
            "rotor_zero",
            "rotor_max",
        ]
    )

    input_t_raw = input_data["t_ros"]

    input_u_raw = matrix_from_dict(
        input_data,
        INPUT_COLUMNS
    )

    input_t, input_u, input_duplicates = (
        sort_keep_last(
            input_t_raw,
            input_u_raw
        )
    )

    # ------------------------------------------------------------------
    # Excitation phases
    # ------------------------------------------------------------------

    exc_data = read_numeric_csv(
        exc_path,
        [
            "t_ros",
            "phase_code",
        ]
    )

    exc_t = exc_data["t_ros"]
    exc_phase = exc_data["phase_code"]

    # Phase 1:
    # stable pre-excitation hover
    pre_t0, pre_t1 = phase_window(
        exc_t,
        exc_phase,
        1
    )

    # Phase 2:
    # active identification excitation
    act_t0, act_t1 = phase_window(
        exc_t,
        exc_phase,
        2
    )

    # ------------------------------------------------------------------
    # Synchronize stable pre-hover window
    # ------------------------------------------------------------------

    (
        t_pre,
        x_pre,
        u_pre
    ) = synchronize_window(
        state_t,
        state_x,
        input_t,
        input_u,
        pre_t0,
        pre_t1,
        ts
    )

    # Empirical operating point.
    x_trim = np.mean(
        x_pre,
        axis=0
    )

    u_trim_phys = np.mean(
        u_pre,
        axis=0
    )

    # MPC-coordinate equivalent of the same empirical trim.
    u_trim_nominal = (
        u_trim_phys
        - U_E
    )

    # ------------------------------------------------------------------
    # Synchronize active identification window
    # ------------------------------------------------------------------

    (
        t_active,
        x_phys,
        u_phys
    ) = synchronize_window(
        state_t,
        state_x,
        input_t,
        input_u,
        act_t0,
        act_t1,
        ts
    )

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    # Nominal MPC input coordinates:
    #
    #   [T - T_hover, tau_phi, tau_theta, tau_psi]
    #
    u_nominal = (
        u_phys
        - U_E
    )

    # Per-run centered coordinates used for PEM regression.
    x_centered = (
        x_phys
        - x_trim
    )

    u_centered = (
        u_phys
        - u_trim_phys
    )

    # Equivalent relation:
    #
    # u_centered
    #   = u_nominal - u_trim_nominal
    #
    if not np.allclose(
        u_centered,
        u_nominal - u_trim_nominal,
        atol=1.0e-12,
        rtol=0.0
    ):
        raise RuntimeError(
            "Input coordinate consistency failure."
        )

    # ------------------------------------------------------------------
    # Validate synchronized data
    # ------------------------------------------------------------------

    validate_uniform_time(
        t_pre,
        ts
    )

    validate_uniform_time(
        t_active,
        ts
    )

    for name, arr in [
        ("x_pre", x_pre),
        ("u_pre", u_pre),
        ("x_phys", x_phys),
        ("u_phys", u_phys),
        ("x_centered", x_centered),
        ("u_centered", u_centered),
    ]:
        validate_finite(
            name,
            arr
        )

    if x_centered.shape[1] != NX:
        raise RuntimeError(
            "Invalid state dimension."
        )

    if u_centered.shape[1] != NU:
        raise RuntimeError(
            "Invalid input dimension."
        )

    if len(t_active) < 3:
        raise RuntimeError(
            "Too few active samples."
        )

    # ------------------------------------------------------------------
    # Build transitions INSIDE this run
    # ------------------------------------------------------------------

    x_k = x_centered[:-1, :]
    u_k = u_centered[:-1, :]
    x_kp1 = x_centered[1:, :]

    t_k = t_active[:-1]

    if not (
        len(x_k)
        == len(u_k)
        == len(x_kp1)
        == len(t_k)
    ):
        raise RuntimeError(
            "Transition length mismatch."
        )

    # ------------------------------------------------------------------
    # Rotor feasibility sanity check in raw active interval
    # ------------------------------------------------------------------

    raw_input_mask = (
        (input_data["t_ros"] >= act_t0)
        &
        (input_data["t_ros"] <= act_t1)
    )

    rotor_zero_count = int(
        np.sum(
            input_data["rotor_zero"][raw_input_mask]
            > 0.5
        )
    )

    rotor_max_count = int(
        np.sum(
            input_data["rotor_max"][raw_input_mask]
            > 0.5
        )
    )

    # ------------------------------------------------------------------
    # Save per-run result
    # ------------------------------------------------------------------

    out_dir = os.path.join(
        output_root,
        run_name
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    out_npz = os.path.join(
        out_dir,
        "processed.npz"
    )

    np.savez_compressed(
        out_npz,

        Ts=np.asarray(
            ts,
            dtype=float
        ),

        state_columns=np.asarray(
            STATE_COLUMNS
        ),

        input_columns=np.asarray(
            INPUT_COLUMNS
        ),

        t_pre=t_pre,

        X_pre=x_pre,

        U_pre_phys=u_pre,

        x_trim=x_trim,

        u_trim_phys=u_trim_phys,

        u_trim_nominal=u_trim_nominal,

        t_active=t_active,

        X_phys=x_phys,

        U_phys=u_phys,

        U_nominal=u_nominal,

        X_centered=x_centered,

        U_centered=u_centered,

        t_k=t_k,

        X_k=x_k,

        U_k=u_k,

        X_kp1=x_kp1,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "run_name": run_name,
        "role": role,

        "Ts": float(ts),

        "state_order": STATE_COLUMNS,
        "input_order": INPUT_COLUMNS,

        "phase1_prehover_raw": {
            "t0": float(pre_t0),
            "t1": float(pre_t1),
            "duration": float(
                pre_t1 - pre_t0
            ),
        },

        "phase2_active_raw": {
            "t0": float(act_t0),
            "t1": float(act_t1),
            "duration": float(
                act_t1 - act_t0
            ),
        },

        "synchronized": {
            "prehover_samples": int(
                len(t_pre)
            ),
            "active_samples": int(
                len(t_active)
            ),
            "transitions": int(
                len(x_k)
            ),
        },

        "duplicates_removed": {
            "state": int(
                state_duplicates
            ),
            "input": int(
                input_duplicates
            ),
        },

        "x_trim": {
            STATE_COLUMNS[i]:
                float(x_trim[i])
            for i in range(NX)
        },

        "u_trim_phys": {
            INPUT_COLUMNS[i]:
                float(u_trim_phys[i])
            for i in range(NU)
        },

        "u_trim_nominal": {
            INPUT_COLUMNS[i]:
                float(u_trim_nominal[i])
            for i in range(NU)
        },

        "rotor_feasibility": {
            "rotor_zero_count":
                rotor_zero_count,
            "rotor_max_count":
                rotor_max_count,
        },

        "processed_file":
            out_npz,
    }

    summary_path = os.path.join(
        out_dir,
        "summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # Return arrays needed for group concatenation.
    return {
        "name": run_name,
        "role": role,
        "X_k": x_k,
        "U_k": u_k,
        "X_kp1": x_kp1,
        "summary": summary,
    }


# ======================================================================
# Dataset concatenation
# ======================================================================

def save_group_dataset(
        results,
        output_path,
        group_name,
        ts):
    """
    Concatenate already-built per-run transitions.

    This preserves run boundaries because X_k/U_k/X_kp1
    were constructed separately inside each run.
    """

    if not results:
        raise RuntimeError(
            "Empty {} dataset.".format(
                group_name
            )
        )

    x_k = np.vstack([
        r["X_k"]
        for r in results
    ])

    u_k = np.vstack([
        r["U_k"]
        for r in results
    ])

    x_kp1 = np.vstack([
        r["X_kp1"]
        for r in results
    ])

    run_names = [
        r["name"]
        for r in results
    ]

    segment_lengths = np.asarray([
        len(r["X_k"])
        for r in results
    ], dtype=int)

    run_index = np.concatenate([
        np.full(
            len(r["X_k"]),
            i,
            dtype=int
        )
        for i, r in enumerate(results)
    ])

    validate_finite(
        "{} X_k".format(group_name),
        x_k
    )

    validate_finite(
        "{} U_k".format(group_name),
        u_k
    )

    validate_finite(
        "{} X_kp1".format(group_name),
        x_kp1
    )

    np.savez_compressed(
        output_path,

        group=np.asarray(
            group_name
        ),

        Ts=np.asarray(
            ts,
            dtype=float
        ),

        state_columns=np.asarray(
            STATE_COLUMNS
        ),

        input_columns=np.asarray(
            INPUT_COLUMNS
        ),

        run_names=np.asarray(
            run_names
        ),

        segment_lengths=segment_lengths,

        run_index=run_index,

        X_k=x_k,

        U_k=u_k,

        X_kp1=x_kp1,
    )

    return {
        "group": group_name,
        "runs": run_names,
        "segment_lengths":
            segment_lengths.tolist(),
        "total_transitions":
            int(len(x_k)),
        "X_k_shape":
            list(x_k.shape),
        "U_k_shape":
            list(u_k.shape),
        "X_kp1_shape":
            list(x_kp1.shape),
        "file":
            output_path,
    }


# ======================================================================
# Main
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "DD21 PEM offline preprocessing."
        )
    )

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    package_dir = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            ".."
        )
    )

    default_data_root = os.path.join(
        package_dir,
        "data",
        "pem_identification"
    )

    default_output_root = os.path.join(
        default_data_root,
        "processed"
    )

    parser.add_argument(
        "--data-root",
        default=default_data_root
    )

    parser.add_argument(
        "--output-root",
        default=default_output_root
    )

    parser.add_argument(
        "--ts",
        type=float,
        default=TS_DEFAULT
    )

    args = parser.parse_args()

    data_root = os.path.abspath(
        args.data_root
    )

    output_root = os.path.abspath(
        args.output_root
    )

    ts = float(
        args.ts
    )

    if ts <= 0.0:
        raise RuntimeError(
            "Ts must be positive."
        )

    os.makedirs(
        output_root,
        exist_ok=True
    )

    print("=" * 78)
    print(" DD21 PEM OFFLINE PREPROCESSING")
    print("=" * 78)

    print(
        "data root   :",
        data_root
    )

    print(
        "output root :",
        output_root
    )

    print(
        "Ts          :",
        ts
    )

    print(
        "state dim   :",
        NX
    )

    print(
        "input dim   :",
        NU
    )

    print()

    train_results = []
    val_results = []

    all_summaries = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    print(
        "[1] TRAINING RUNS"
    )

    print("-" * 78)

    for run_name in TRAIN_RUNS:

        print(
            "Processing:",
            run_name
        )

        result = process_run(
            run_name=run_name,
            role="training",
            data_root=data_root,
            output_root=output_root,
            ts=ts
        )

        train_results.append(
            result
        )

        all_summaries.append(
            result["summary"]
        )

        s = result["summary"]

        print(
            "  pre-hover samples :",
            s["synchronized"][
                "prehover_samples"
            ]
        )

        print(
            "  active samples    :",
            s["synchronized"][
                "active_samples"
            ]
        )

        print(
            "  transitions       :",
            s["synchronized"][
                "transitions"
            ]
        )

        print(
            "  trim T            :",
            "{:.9f}".format(
                s["u_trim_phys"]["T"]
            )
        )

        print(
            "  trim tau_phi      :",
            "{:+.9f}".format(
                s["u_trim_phys"][
                    "tau_phi"
                ]
            )
        )

        print(
            "  trim tau_theta    :",
            "{:+.9f}".format(
                s["u_trim_phys"][
                    "tau_theta"
                ]
            )
        )

        print(
            "  trim tau_psi      :",
            "{:+.9f}".format(
                s["u_trim_phys"][
                    "tau_psi"
                ]
            )
        )

        print(
            "  rotor zero/max    :",
            "{}/{}".format(
                s["rotor_feasibility"][
                    "rotor_zero_count"
                ],
                s["rotor_feasibility"][
                    "rotor_max_count"
                ]
            )
        )

        print(
            "  [PASS]"
        )

        print()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    print(
        "[2] HELD-OUT VALIDATION RUNS"
    )

    print("-" * 78)

    for run_name in VALIDATION_RUNS:

        print(
            "Processing:",
            run_name
        )

        result = process_run(
            run_name=run_name,
            role="validation",
            data_root=data_root,
            output_root=output_root,
            ts=ts
        )

        val_results.append(
            result
        )

        all_summaries.append(
            result["summary"]
        )

        s = result["summary"]

        print(
            "  pre-hover samples :",
            s["synchronized"][
                "prehover_samples"
            ]
        )

        print(
            "  active samples    :",
            s["synchronized"][
                "active_samples"
            ]
        )

        print(
            "  transitions       :",
            s["synchronized"][
                "transitions"
            ]
        )

        print(
            "  trim T            :",
            "{:.9f}".format(
                s["u_trim_phys"]["T"]
            )
        )

        print(
            "  trim tau_phi      :",
            "{:+.9f}".format(
                s["u_trim_phys"][
                    "tau_phi"
                ]
            )
        )

        print(
            "  trim tau_theta    :",
            "{:+.9f}".format(
                s["u_trim_phys"][
                    "tau_theta"
                ]
            )
        )

        print(
            "  trim tau_psi      :",
            "{:+.9f}".format(
                s["u_trim_phys"][
                    "tau_psi"
                ]
            )
        )

        print(
            "  rotor zero/max    :",
            "{}/{}".format(
                s["rotor_feasibility"][
                    "rotor_zero_count"
                ],
                s["rotor_feasibility"][
                    "rotor_max_count"
                ]
            )
        )

        print(
            "  [PASS]"
        )

        print()

    # ------------------------------------------------------------------
    # Save combined groups
    # ------------------------------------------------------------------

    print(
        "[3] BUILD GROUP DATASETS"
    )

    print("-" * 78)

    train_path = os.path.join(
        output_root,
        "pem_train_dataset.npz"
    )

    val_path = os.path.join(
        output_root,
        "pem_val_dataset.npz"
    )

    train_group = save_group_dataset(
        train_results,
        train_path,
        "training",
        ts
    )

    val_group = save_group_dataset(
        val_results,
        val_path,
        "validation",
        ts
    )

    print(
        "Training X_k   :",
        tuple(
            train_group[
                "X_k_shape"
            ]
        )
    )

    print(
        "Training U_k   :",
        tuple(
            train_group[
                "U_k_shape"
            ]
        )
    )

    print(
        "Training X_kp1 :",
        tuple(
            train_group[
                "X_kp1_shape"
            ]
        )
    )

    print()

    print(
        "Validation X_k   :",
        tuple(
            val_group[
                "X_k_shape"
            ]
        )
    )

    print(
        "Validation U_k   :",
        tuple(
            val_group[
                "U_k_shape"
            ]
        )
    )

    print(
        "Validation X_kp1 :",
        tuple(
            val_group[
                "X_kp1_shape"
            ]
        )
    )

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------

    global_summary = {
        "model": (
            "dx[k+1] = A_PEM dx[k] "
            "+ B_PEM du[k]"
        ),

        "Ts": ts,

        "state_order":
            STATE_COLUMNS,

        "input_order":
            INPUT_COLUMNS,

        "T_hover_nominal":
            T_HOVER,

        "trim_strategy": (
            "per-run empirical operating point "
            "from phase-1 stable pre-hover"
        ),

        "state_synchronization":
            "linear interpolation",

        "input_synchronization":
            "zero-order hold",

        "transition_policy": (
            "transitions formed inside each "
            "run before concatenation"
        ),

        "training":
            train_group,

        "validation":
            val_group,

        "runs":
            all_summaries,
    }

    summary_path = os.path.join(
        output_root,
        "preprocess_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            global_summary,
            f,
            indent=2
        )

    print()

    print("=" * 78)
    print(" PEM PREPROCESSING COMPLETE")
    print("=" * 78)

    print(
        "Training transitions  :",
        train_group[
            "total_transitions"
        ]
    )

    print(
        "Validation transitions:",
        val_group[
            "total_transitions"
        ]
    )

    print()

    print(
        "Training file:"
    )

    print(
        train_path
    )

    print()

    print(
        "Validation file:"
    )

    print(
        val_path
    )

    print()

    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()

    print(
        "[PASS] 100 Hz synchronization"
    )

    print(
        "[PASS] state interpolation"
    )

    print(
        "[PASS] input ZOH"
    )

    print(
        "[PASS] per-run hover centering"
    )

    print(
        "[PASS] run boundaries preserved"
    )

    print(
        "[PASS] training / validation separation preserved"
    )


if __name__ == "__main__":
    main()
