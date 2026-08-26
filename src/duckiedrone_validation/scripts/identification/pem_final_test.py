#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_final_test.py

FINAL untouched test of DD21 PEM candidates.

Models were completely locked BEFORE this dataset was evaluated.

Compared models
---------------
1. FULL_OLS
2. LOCKED_IV_PQ

Final untouched dataset
------------------------
pem_test_mimo_01

Primary evaluation endpoint
---------------------------
Rolling trajectory prediction at:

    H = 20 samples
    Ts = 0.01 s
    horizon = 0.20 s

This corresponds to the current MPC prediction horizon.

No estimation, tuning, delay selection, structural modification,
or parameter refinement is performed in this script.

Author: Abdallah GHOUL
2026
"""

import hashlib
import json
import os

import numpy as np


NX = 12

STATE_NAMES = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]

HORIZONS = [
    1,
    5,
    10,
    20,
    100,
]


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def fit_percent(y, yh):

    denom = np.linalg.norm(
        y - np.mean(y)
    )

    if denom <= 1.0e-15:
        return np.nan

    return 100.0 * (
        1.0
        - np.linalg.norm(
            y - yh
        ) / denom
    )


def metrics(Y, Yh):

    error = (
        Y - Yh
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2,
            axis=0
        )
    )

    mae = np.mean(
        np.abs(error),
        axis=0
    )

    maxae = np.max(
        np.abs(error),
        axis=0
    )

    fit = np.asarray([
        fit_percent(
            Y[:, j],
            Yh[:, j]
        )
        for j in range(NX)
    ])

    return {
        "fit": fit,
        "rmse": rmse,
        "mae": mae,
        "maxae": maxae,
    }


def rolling_terminal(
        A,
        B,
        X,
        U,
        H):
    """
    Exactly H-step-ahead prediction from each
    possible measured prediction origin.
    """

    truth = []
    pred = []

    n = len(X)

    for k0 in range(
        0,
        n - H
    ):

        xh = X[
            k0,
            :
        ].copy()

        for j in range(H):

            k = (
                k0
                + j
            )

            xh = (
                A @ xh
                + B @ U[k, :]
            )

        truth.append(
            X[
                k0 + H,
                :
            ].copy()
        )

        pred.append(
            xh.copy()
        )

    Y = np.asarray(
        truth
    )

    Yh = np.asarray(
        pred
    )

    return (
        Y,
        Yh,
        metrics(
            Y,
            Yh
        )
    )


def rolling_trajectory(
        A,
        B,
        X,
        U,
        H):
    """
    Pool prediction steps 1...H over every
    measured receding-horizon prediction origin.
    """

    truth = []
    pred = []

    n = len(X)

    for k0 in range(
        0,
        n - H
    ):

        xh = X[
            k0,
            :
        ].copy()

        for j in range(H):

            k = (
                k0
                + j
            )

            xh = (
                A @ xh
                + B @ U[k, :]
            )

            truth.append(
                X[
                    k + 1,
                    :
                ].copy()
            )

            pred.append(
                xh.copy()
            )

    Y = np.asarray(
        truth
    )

    Yh = np.asarray(
        pred
    )

    return (
        Y,
        Yh,
        metrics(
            Y,
            Yh
        )
    )


def free_run(
        A,
        B,
        X,
        U):
    """
    Entire 15-second open-loop simulation.

    Reported for completeness only.
    It is NOT the primary MPC-model selection criterion.
    """

    Y = X[
        1:,
        :
    ]

    Yh = np.empty_like(
        Y
    )

    xh = X[
        0,
        :
    ].copy()

    for k in range(
        len(X) - 1
    ):

        xh = (
            A @ xh
            + B @ U[k, :]
        )

        Yh[
            k,
            :
        ] = xh

    return (
        Y,
        Yh,
        metrics(
            Y,
            Yh
        )
    )


def print_state_metrics(
        title,
        m):

    print()
    print(title)
    print("-" * 92)

    print(
        "state       FIT [%]          RMSE"
        "             MAE"
    )

    for j, name in enumerate(
        STATE_NAMES
    ):

        print(
            f"{name:8s}"
            f" {m['fit'][j]:12.3f}"
            f" {m['rmse'][j]:14.6e}"
            f" {m['mae'][j]:14.6e}"
        )


def main():

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

    root = os.path.join(
        package_dir,
        "data",
        "pem_identification",
        "processed"
    )

    full_path = os.path.join(
        root,
        "pem_candidate_model.npz"
    )

    locked_path = os.path.join(
        root,
        "pem_locked_iv_pq_candidate.npz"
    )

    test_path = os.path.join(
        root,
        "pem_test_mimo_01",
        "processed.npz"
    )

    for path in [
        full_path,
        locked_path,
        test_path,
    ]:

        if not os.path.isfile(path):
            raise RuntimeError(
                "Missing file:\n{}".format(
                    path
                )
            )

    # --------------------------------------------------------------
    # Record locked-candidate integrity hash BEFORE evaluation.
    # --------------------------------------------------------------

    locked_hash_before = sha256_file(
        locked_path
    )

    # --------------------------------------------------------------
    # Load models
    # --------------------------------------------------------------

    full = np.load(
        full_path,
        allow_pickle=False
    )

    locked = np.load(
        locked_path,
        allow_pickle=False
    )

    models = {
        "FULL_OLS": (
            np.asarray(
                full["A"],
                dtype=float
            ),
            np.asarray(
                full["B"],
                dtype=float
            ),
        ),

        "LOCKED_IV_PQ": (
            np.asarray(
                locked["A"],
                dtype=float
            ),
            np.asarray(
                locked["B"],
                dtype=float
            ),
        ),
    }

    # --------------------------------------------------------------
    # Load FINAL untouched test only.
    # --------------------------------------------------------------

    test = np.load(
        test_path,
        allow_pickle=False
    )

    X = np.asarray(
        test["X_centered"],
        dtype=float
    )

    U = np.asarray(
        test["U_centered"],
        dtype=float
    )

    Ts = float(
        test["Ts"]
    )

    if len(X) != len(U):
        raise RuntimeError(
            "Final-test X/U length mismatch."
        )

    if X.shape[1] != 12:
        raise RuntimeError(
            "Invalid final-test state dimension."
        )

    if U.shape[1] != 4:
        raise RuntimeError(
            "Invalid final-test input dimension."
        )

    print("=" * 108)
    print("DD21 PEM — FINAL UNTOUCHED TEST")
    print("=" * 108)

    print()
    print(
        "dataset : pem_test_mimo_01"
    )

    print(
        "samples :",
        len(X)
    )

    print(
        "Ts      :",
        Ts
    )

    print(
        "MPC Np  : 20"
    )

    print(
        "MPC horizon:",
        20 * Ts,
        "s"
    )

    print()
    print(
        "LOCKED candidate SHA256:"
    )

    print(
        locked_hash_before
    )

    results = {}

    # --------------------------------------------------------------
    # Evaluate both PRE-LOCKED models identically.
    # --------------------------------------------------------------

    for model_name, (
            A,
            B) in models.items():

        if A.shape != (
                12,
                12):

            raise RuntimeError(
                "Invalid A shape for "
                + model_name
            )

        if B.shape != (
                12,
                4):

            raise RuntimeError(
                "Invalid B shape for "
                + model_name
            )

        rho = float(
            np.max(
                np.abs(
                    np.linalg.eigvals(A)
                )
            )
        )

        print()
        print("=" * 108)
        print(model_name)
        print("=" * 108)

        print(
            "spectral radius :",
            rho
        )

        model_results = {
            "spectral_radius":
                rho,
            "horizons": {},
        }

        for H in HORIZONS:

            (
                Yterminal,
                Yhterminal,
                terminal
            ) = rolling_terminal(
                A,
                B,
                X,
                U,
                H
            )

            (
                Ytraj,
                Yhtraj,
                trajectory
            ) = rolling_trajectory(
                A,
                B,
                X,
                U,
                H
            )

            model_results[
                "horizons"
            ][H] = {
                "terminal":
                    terminal,
                "trajectory":
                    trajectory,
            }

            print()
            print("-" * 108)

            print(
                f"H = {H:3d} "
                f"({H * Ts:.2f} s)"
            )

            print(
                "terminal mean FIT   :",
                np.nanmean(
                    terminal["fit"]
                )
            )

            print(
                "trajectory mean FIT :",
                np.nanmean(
                    trajectory["fit"]
                )
            )

            if H == 20:

                print_state_metrics(
                    "H=20 ROLLING TRAJECTORY — STATE METRICS",
                    trajectory
                )

                print()
                print(
                    "H20 terminal:"
                )

                print(
                    "  phi   :",
                    terminal["fit"][3]
                )

                print(
                    "  theta :",
                    terminal["fit"][4]
                )

                print(
                    "  p     :",
                    terminal["fit"][9]
                )

                print(
                    "  q     :",
                    terminal["fit"][10]
                )

        (
            Yfree,
            Yhfree,
            free
        ) = free_run(
            A,
            B,
            X,
            U
        )

        model_results[
            "free"
        ] = free

        print()
        print(
            "15 s FREE-RUN mean FIT:",
            np.nanmean(
                free["fit"]
            )
        )

        print(
            "(reported only; not primary selection endpoint)"
        )

        results[
            model_name
        ] = model_results

    # --------------------------------------------------------------
    # Compact final comparison
    # --------------------------------------------------------------

    print()
    print("=" * 118)
    print("FINAL COMPACT COMPARISON")
    print("=" * 118)

    print(
        "model          "
        "H1 traj     H10 traj    H20 traj    "
        "H20 term    H100 traj    FREE"
    )

    for name in [
        "FULL_OLS",
        "LOCKED_IV_PQ",
    ]:

        r = results[
            name
        ]

        print(
            f"{name:14s}"
            f" "
            f"{np.nanmean(r['horizons'][1]['trajectory']['fit']):9.3f}"
            f" "
            f"{np.nanmean(r['horizons'][10]['trajectory']['fit']):11.3f}"
            f" "
            f"{np.nanmean(r['horizons'][20]['trajectory']['fit']):11.3f}"
            f" "
            f"{np.nanmean(r['horizons'][20]['terminal']['fit']):10.3f}"
            f" "
            f"{np.nanmean(r['horizons'][100]['trajectory']['fit']):12.3f}"
            f" "
            f"{np.nanmean(r['free']['fit']):9.3f}"
        )

    # --------------------------------------------------------------
    # Pre-declared decision diagnostics
    # --------------------------------------------------------------

    base = results[
        "FULL_OLS"
    ]

    iv = results[
        "LOCKED_IV_PQ"
    ]

    base_h1 = np.nanmean(
        base["horizons"][1][
            "trajectory"
        ]["fit"]
    )

    iv_h1 = np.nanmean(
        iv["horizons"][1][
            "trajectory"
        ]["fit"]
    )

    base_h20 = np.nanmean(
        base["horizons"][20][
            "trajectory"
        ]["fit"]
    )

    iv_h20 = np.nanmean(
        iv["horizons"][20][
            "trajectory"
        ]["fit"]
    )

    bfit = base[
        "horizons"
    ][20][
        "trajectory"
    ]["fit"]

    ifit = iv[
        "horizons"
    ][20][
        "trajectory"
    ]["fit"]

    primary_improvement = (
        iv_h20
        > base_h20
    )

    p_improved = (
        ifit[9]
        > bfit[9]
    )

    q_improved = (
        ifit[10]
        > bfit[10]
    )

    phi_not_worse = (
        ifit[3]
        >= bfit[3]
    )

    theta_not_worse = (
        ifit[4]
        >= bfit[4]
    )

    one_step_not_materially_worse = (
        iv_h1
        >= base_h1 - 1.0
    )

    final_pass = (
        primary_improvement
        and p_improved
        and q_improved
        and phi_not_worse
        and theta_not_worse
        and one_step_not_materially_worse
    )

    print()
    print("=" * 118)
    print("PRE-DECLARED FINAL-TEST CHECK")
    print("=" * 118)

    print(
        "Primary H20 trajectory improvement :",
        primary_improvement
    )

    print(
        "H20 p improved                    :",
        p_improved
    )

    print(
        "H20 q improved                    :",
        q_improved
    )

    print(
        "H20 phi not worse                 :",
        phi_not_worse
    )

    print(
        "H20 theta not worse               :",
        theta_not_worse
    )

    print(
        "H1 mean degradation <= 1 point    :",
        one_step_not_materially_worse
    )

    print()
    print(
        "H20 mean change [percentage points] :",
        iv_h20 - base_h20
    )

    print(
        "p H20 change                        :",
        ifit[9] - bfit[9]
    )

    print(
        "q H20 change                        :",
        ifit[10] - bfit[10]
    )

    print(
        "phi H20 change                      :",
        ifit[3] - bfit[3]
    )

    print(
        "theta H20 change                    :",
        ifit[4] - bfit[4]
    )

    print()
    print(
        "FINAL PRE-DECLARED PASS:",
        final_pass
    )

    # --------------------------------------------------------------
    # Verify locked candidate file was not modified.
    # --------------------------------------------------------------

    locked_hash_after = sha256_file(
        locked_path
    )

    if (
        locked_hash_after
        != locked_hash_before
    ):
        raise RuntimeError(
            "LOCKED candidate file changed "
            "during final test."
        )

    print()
    print(
        "Locked candidate hash unchanged:",
        True
    )

    # --------------------------------------------------------------
    # Save evaluation results only.
    # NO MODEL MODIFICATION.
    # --------------------------------------------------------------

    output_npz = os.path.join(
        root,
        "pem_final_test_results.npz"
    )

    np.savez_compressed(
        output_npz,

        locked_candidate_sha256=
            np.asarray(
                locked_hash_before
            ),

        Ts=np.asarray(
            Ts,
            dtype=float
        ),

        full_h20_trajectory_fit=
            base[
                "horizons"
            ][20][
                "trajectory"
            ]["fit"],

        iv_h20_trajectory_fit=
            iv[
                "horizons"
            ][20][
                "trajectory"
            ]["fit"],

        full_h20_terminal_fit=
            base[
                "horizons"
            ][20][
                "terminal"
            ]["fit"],

        iv_h20_terminal_fit=
            iv[
                "horizons"
            ][20][
                "terminal"
            ]["fit"],

        final_pass=np.asarray(
            final_pass,
            dtype=bool
        ),
    )

    summary = {
        "dataset":
            "pem_test_mimo_01",

        "dataset_role":
            "FINAL UNTOUCHED TEST",

        "model_tuning_on_final_test":
            False,

        "locked_candidate_sha256":
            locked_hash_before,

        "primary_endpoint":
            "rolling trajectory mean FIT at H=20",

        "FULL_OLS_H20_mean_fit_percent":
            float(
                base_h20
            ),

        "LOCKED_IV_PQ_H20_mean_fit_percent":
            float(
                iv_h20
            ),

        "H20_improvement_percentage_points":
            float(
                iv_h20
                - base_h20
            ),

        "final_predeclared_pass":
            bool(
                final_pass
            ),

        "result_file":
            output_npz,
    }

    summary_path = os.path.join(
        root,
        "pem_final_test_summary.json"
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

    print()
    print("Results:")
    print(output_npz)

    print()
    print("Summary:")
    print(summary_path)

    print()
    print(
        "[RULE] No parameter tuning is permitted "
        "after this final test."
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
