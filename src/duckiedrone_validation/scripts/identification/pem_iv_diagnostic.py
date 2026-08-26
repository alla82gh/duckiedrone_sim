#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_iv_diagnostic.py

Instrumental-variable diagnostic for DD21 PEM rotational dynamics.

Purpose
-------
Check whether closed-loop correlation biases the OLS estimates of:

    vz[k+1] = a_vz vz[k] + b_T       dT[k]
    p [k+1] = a_p  p [k] + b_phi     tau_phi[k]
    q [k+1] = a_q  q [k] + b_theta   tau_theta[k]
    r [k+1] = a_r  r [k] + b_psi     tau_psi[k]

The exogenous PRBS excitation is used as an instrument.

Training runs ONLY are used for IV parameter estimation.

No model file is modified.

Author: Abdallah GHOUL
2026
"""

import csv
import os

import numpy as np


TRAIN_RUNS = [
    "pem_train_thrust_01",
    "pem_train_roll_01",
    "pem_train_pitch_01",
    "pem_train_yaw_01",
    "pem_train_mimo_01",
]


CHANNELS = [
    # label, state index, input index, excitation CSV column, physical reference
    ("vz", 8,  0, "delta_T_exc",   0.01 / 0.635),
    ("p",  9,  1, "tau_phi_exc",   0.01 / 0.0015),
    ("q",  10, 2, "tau_theta_exc", 0.01 / 0.0017),
    ("r",  11, 3, "tau_psi_exc",   0.01 / 0.0030),
]


STATE_NAMES = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]


def read_excitation(path):

    cols = [
        "t_ros",
        "phase_code",
        "delta_T_exc",
        "tau_phi_exc",
        "tau_theta_exc",
        "tau_psi_exc",
    ]

    out = {
        c: []
        for c in cols
    }

    with open(path, newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if int(
                round(
                    float(
                        row["phase_code"]
                    )
                )
            ) != 2:
                continue

            for c in cols:

                out[c].append(
                    float(
                        row[c]
                    )
                )

    return {
        c: np.asarray(
            out[c],
            dtype=float
        )
        for c in cols
    }


def zoh(t_source, y_source, t_target):

    order = np.argsort(
        t_source,
        kind="stable"
    )

    t_source = t_source[order]
    y_source = y_source[order]

    idx = (
        np.searchsorted(
            t_source,
            t_target,
            side="right"
        )
        - 1
    )

    idx = np.clip(
        idx,
        0,
        len(t_source) - 1
    )

    return y_source[idx]


def load_run(
        processed_root,
        raw_root,
        run_name):

    processed_path = os.path.join(
        processed_root,
        run_name,
        "processed.npz"
    )

    excitation_path = os.path.join(
        raw_root,
        run_name,
        "excitation_raw.csv"
    )

    d = np.load(
        processed_path,
        allow_pickle=False
    )

    t = np.asarray(
        d["t_active"],
        dtype=float
    )

    X = np.asarray(
        d["X_centered"],
        dtype=float
    )

    U = np.asarray(
        d["U_centered"],
        dtype=float
    )

    exc = read_excitation(
        excitation_path
    )

    Zexc = np.column_stack([
        zoh(
            exc["t_ros"],
            exc["delta_T_exc"],
            t
        ),
        zoh(
            exc["t_ros"],
            exc["tau_phi_exc"],
            t
        ),
        zoh(
            exc["t_ros"],
            exc["tau_theta_exc"],
            t
        ),
        zoh(
            exc["t_ros"],
            exc["tau_psi_exc"],
            t
        ),
    ])

    return X, U, Zexc


def ols_scalar(x, u, y):

    R = np.column_stack([
        x,
        u
    ])

    theta, _, _, _ = np.linalg.lstsq(
        R,
        y,
        rcond=None
    )

    return theta


def iv_scalar(
        x_prev,
        z_exc,
        x,
        u,
        y):
    """
    2SLS:

    regressors:
        R = [x[k], u[k]]

    instruments:
        W = [x[k-1], PRBS[k]]

    Stage 1:
        Rhat = projection of R onto W

    Stage 2:
        y ~ Rhat
    """

    R = np.column_stack([
        x,
        u
    ])

    W = np.column_stack([
        x_prev,
        z_exc
    ])

    # Center instruments and regressors for numerical robustness.
    # No intercept is introduced into the final centered model.
    R_mean = np.mean(
        R,
        axis=0,
        keepdims=True
    )

    W_mean = np.mean(
        W,
        axis=0,
        keepdims=True
    )

    Rc = R - R_mean
    Wc = W - W_mean

    # First-stage projection.
    Pi, _, _, _ = np.linalg.lstsq(
        Wc,
        Rc,
        rcond=None
    )

    Rhat_c = (
        Wc @ Pi
    )

    # Restore regressor mean.
    Rhat = (
        Rhat_c
        + R_mean
    )

    theta, _, _, _ = np.linalg.lstsq(
        Rhat,
        y,
        rcond=None
    )

    first_stage_rank = int(
        np.linalg.matrix_rank(
            Rhat_c
        )
    )

    # Instrument/input correlation is useful
    # as a simple first-stage strength diagnostic.
    corr_zu = np.corrcoef(
        z_exc,
        u
    )[0, 1]

    return (
        theta,
        first_stage_rank,
        corr_zu
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

    raw_root = os.path.join(
        package_dir,
        "data",
        "pem_identification"
    )

    processed_root = os.path.join(
        raw_root,
        "processed"
    )

    print("=" * 100)
    print("DD21 PEM — INSTRUMENTAL VARIABLE DIAGNOSTIC")
    print("=" * 100)

    print()
    print(
        "Training runs only:"
    )

    for run in TRAIN_RUNS:
        print("  ", run)

    loaded = {}

    for run in TRAIN_RUNS:

        loaded[run] = load_run(
            processed_root,
            raw_root,
            run
        )

    for (
            label,
            state_idx,
            input_idx,
            exc_col,
            physical_b) in CHANNELS:

        print()
        print("=" * 100)

        print(
            "CHANNEL:",
            label
        )

        print("=" * 100)

        x_prev_all = []
        x_all = []
        u_all = []
        y_all = []
        z_all = []

        for run in TRAIN_RUNS:

            X, U, Zexc = loaded[
                run
            ]

            # Need k-1, k, k+1.
            #
            # k = 1 ... N-2
            x_prev = X[
                :-2,
                state_idx
            ]

            x = X[
                1:-1,
                state_idx
            ]

            y = X[
                2:,
                state_idx
            ]

            u = U[
                1:-1,
                input_idx
            ]

            z = Zexc[
                1:-1,
                input_idx
            ]

            x_prev_all.append(
                x_prev
            )

            x_all.append(
                x
            )

            u_all.append(
                u
            )

            y_all.append(
                y
            )

            z_all.append(
                z
            )

        x_prev = np.concatenate(
            x_prev_all
        )

        x = np.concatenate(
            x_all
        )

        u = np.concatenate(
            u_all
        )

        y = np.concatenate(
            y_all
        )

        z = np.concatenate(
            z_all
        )

        ols = ols_scalar(
            x,
            u,
            y
        )

        (
            iv,
            iv_rank,
            corr_zu
        ) = iv_scalar(
            x_prev,
            z,
            x,
            u,
            y
        )

        print(
            "samples               :",
            len(y)
        )

        print(
            "instrument column      :",
            exc_col
        )

        print(
            "corr(PRBS, actual u)   :",
            corr_zu
        )

        print(
            "IV first-stage rank    :",
            iv_rank,
            "/ 2"
        )

        print()
        print(
            "OLS:"
        )

        print(
            "  a =",
            ols[0]
        )

        print(
            "  b =",
            ols[1]
        )

        print()
        print(
            "IV / 2SLS:"
        )

        print(
            "  a =",
            iv[0]
        )

        print(
            "  b =",
            iv[1]
        )

        print()
        print(
            "physical reference b  :",
            physical_b
        )

        print(
            "OLS b/reference       :",
            ols[1] / physical_b
        )

        print(
            "IV  b/reference       :",
            iv[1] / physical_b
        )

        # One-step errors using the same training equations.
        yh_ols = (
            ols[0] * x
            + ols[1] * u
        )

        yh_iv = (
            iv[0] * x
            + iv[1] * u
        )

        rmse_ols = np.sqrt(
            np.mean(
                (y - yh_ols) ** 2
            )
        )

        rmse_iv = np.sqrt(
            np.mean(
                (y - yh_iv) ** 2
            )
        )

        print()
        print(
            "training RMSE OLS      :",
            rmse_ols
        )

        print(
            "training RMSE IV       :",
            rmse_iv
        )

    print()
    print("=" * 100)

    print(
        "NOTE: diagnostic only — no model file was modified."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
