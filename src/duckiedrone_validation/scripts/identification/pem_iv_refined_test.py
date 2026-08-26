#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_iv_refined_test.py

Development-validation test of IV-refined DD21 PEM candidates.

Baseline:
    Full 12-state OLS model.

Candidate IV_PQ:
    Keep the full OLS model except replace the p and q equations by
    training-only IV estimates:

        p[k+1] = a_p p[k] + b_p tau_phi[k]
        q[k+1] = a_q q[k] + b_q tau_theta[k]

Candidate IV_PQR:
    Same, but also replace the r equation.

No final model is saved by this script.
pem_val_mimo_01 is used only as DEVELOPMENT validation.

Author: Abdallah GHOUL
2026
"""

import csv
import os
import numpy as np


NX = 12

TRAIN_RUNS = [
    "pem_train_thrust_01",
    "pem_train_roll_01",
    "pem_train_pitch_01",
    "pem_train_yaw_01",
    "pem_train_mimo_01",
]

STATE_NAMES = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]


def read_excitation(path):

    columns = [
        "t_ros",
        "phase_code",
        "delta_T_exc",
        "tau_phi_exc",
        "tau_theta_exc",
        "tau_psi_exc",
    ]

    out = {c: [] for c in columns}

    with open(path, newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            phase = int(
                round(
                    float(row["phase_code"])
                )
            )

            if phase != 2:
                continue

            for c in columns:
                out[c].append(
                    float(row[c])
                )

    return {
        c: np.asarray(v, dtype=float)
        for c, v in out.items()
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


def load_training_run(
        processed_root,
        raw_root,
        run_name):

    d = np.load(
        os.path.join(
            processed_root,
            run_name,
            "processed.npz"
        ),
        allow_pickle=False
    )

    X = np.asarray(
        d["X_centered"],
        dtype=float
    )

    U = np.asarray(
        d["U_centered"],
        dtype=float
    )

    t = np.asarray(
        d["t_active"],
        dtype=float
    )

    exc = read_excitation(
        os.path.join(
            raw_root,
            run_name,
            "excitation_raw.csv"
        )
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


def estimate_iv_channel(
        loaded,
        state_index,
        input_index):

    xp_all = []
    x_all = []
    u_all = []
    y_all = []
    z_all = []

    for run_name in TRAIN_RUNS:

        X, U, Zexc = loaded[
            run_name
        ]

        xp_all.append(
            X[:-2, state_index]
        )

        x_all.append(
            X[1:-1, state_index]
        )

        y_all.append(
            X[2:, state_index]
        )

        u_all.append(
            U[1:-1, input_index]
        )

        z_all.append(
            Zexc[1:-1, input_index]
        )

    xp = np.concatenate(xp_all)
    x = np.concatenate(x_all)
    u = np.concatenate(u_all)
    y = np.concatenate(y_all)
    z = np.concatenate(z_all)

    R = np.column_stack([
        x,
        u
    ])

    W = np.column_stack([
        xp,
        z
    ])

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

    Pi, _, _, _ = np.linalg.lstsq(
        Wc,
        Rc,
        rcond=None
    )

    Rhat = (
        Wc @ Pi
        + R_mean
    )

    theta, _, _, _ = np.linalg.lstsq(
        Rhat,
        y,
        rcond=None
    )

    return theta


def fit_percent(y, yh):

    denom = np.linalg.norm(
        y - np.mean(y)
    )

    if denom <= 1e-15:
        return np.nan

    return 100.0 * (
        1.0
        - np.linalg.norm(y - yh)
        / denom
    )


def metrics(Y, Yh):

    fit = np.asarray([
        fit_percent(
            Y[:, j],
            Yh[:, j]
        )
        for j in range(NX)
    ])

    rmse = np.sqrt(
        np.mean(
            (Y - Yh) ** 2,
            axis=0
        )
    )

    return fit, rmse


def rolling_prediction(
        A,
        B,
        X,
        U,
        H):

    truth = []
    prediction = []

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

            k = k0 + j

            xh = (
                A @ xh
                + B @ U[k, :]
            )

            truth.append(
                X[k + 1, :].copy()
            )

            prediction.append(
                xh.copy()
            )

    return metrics(
        np.asarray(truth),
        np.asarray(prediction)
    )


def free_run(
        A,
        B,
        X,
        U):

    Y = X[1:, :]

    Yh = np.empty_like(Y)

    xh = X[0, :].copy()

    for k in range(
        len(X) - 1
    ):

        xh = (
            A @ xh
            + B @ U[k, :]
        )

        Yh[k, :] = xh

    return metrics(
        Y,
        Yh
    )


def make_sparse_row(
        A,
        B,
        state_row,
        state_col,
        input_col,
        theta):

    A[state_row, :] = 0.0
    B[state_row, :] = 0.0

    A[
        state_row,
        state_col
    ] = theta[0]

    B[
        state_row,
        input_col
    ] = theta[1]


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

    baseline = np.load(
        os.path.join(
            processed_root,
            "pem_candidate_model.npz"
        ),
        allow_pickle=False
    )

    A0 = np.asarray(
        baseline["A"],
        dtype=float
    )

    B0 = np.asarray(
        baseline["B"],
        dtype=float
    )

    loaded = {}

    for run_name in TRAIN_RUNS:

        loaded[run_name] = (
            load_training_run(
                processed_root,
                raw_root,
                run_name
            )
        )

    # --------------------------------------------------------------
    # Training-only IV estimates
    # --------------------------------------------------------------

    theta_p = estimate_iv_channel(
        loaded,
        state_index=9,
        input_index=1
    )

    theta_q = estimate_iv_channel(
        loaded,
        state_index=10,
        input_index=2
    )

    theta_r = estimate_iv_channel(
        loaded,
        state_index=11,
        input_index=3
    )

    print("=" * 100)
    print("DD21 PEM — IV REFINED CANDIDATE TEST")
    print("=" * 100)

    print()
    print("TRAINING-ONLY IV PARAMETERS")

    print(
        "p : a =",
        theta_p[0],
        " b =",
        theta_p[1]
    )

    print(
        "q : a =",
        theta_q[0],
        " b =",
        theta_q[1]
    )

    print(
        "r : a =",
        theta_r[0],
        " b =",
        theta_r[1]
    )

    # --------------------------------------------------------------
    # Candidate 1: baseline full OLS
    # --------------------------------------------------------------

    candidates = {
        "FULL_OLS": (
            A0.copy(),
            B0.copy()
        )
    }

    # --------------------------------------------------------------
    # Candidate 2: IV p/q only
    # --------------------------------------------------------------

    A_pq = A0.copy()
    B_pq = B0.copy()

    make_sparse_row(
        A_pq,
        B_pq,
        9,
        9,
        1,
        theta_p
    )

    make_sparse_row(
        A_pq,
        B_pq,
        10,
        10,
        2,
        theta_q
    )

    candidates[
        "IV_PQ"
    ] = (
        A_pq,
        B_pq
    )

    # --------------------------------------------------------------
    # Candidate 3: IV p/q/r
    # --------------------------------------------------------------

    A_pqr = A_pq.copy()
    B_pqr = B_pq.copy()

    make_sparse_row(
        A_pqr,
        B_pqr,
        11,
        11,
        3,
        theta_r
    )

    candidates[
        "IV_PQR"
    ] = (
        A_pqr,
        B_pqr
    )

    # --------------------------------------------------------------
    # Development validation
    # --------------------------------------------------------------

    val = np.load(
        os.path.join(
            processed_root,
            "pem_val_mimo_01",
            "processed.npz"
        ),
        allow_pickle=False
    )

    Xv = np.asarray(
        val["X_centered"],
        dtype=float
    )

    Uv = np.asarray(
        val["U_centered"],
        dtype=float
    )

    horizons = [
        1,
        5,
        10,
        20,
        100,
    ]

    results = {}

    for name, (
            A,
            B) in candidates.items():

        results[name] = {}

        rho = float(
            np.max(
                np.abs(
                    np.linalg.eigvals(A)
                )
            )
        )

        print()
        print("=" * 100)
        print(name)
        print("=" * 100)

        print(
            "spectral radius :",
            rho
        )

        for H in horizons:

            fit, rmse = rolling_prediction(
                A,
                B,
                Xv,
                Uv,
                H
            )

            results[name][H] = fit

            print()
            print(
                f"H={H:3d} "
                f"({H*0.01:.2f}s)"
            )

            print(
                "mean FIT :",
                np.nanmean(fit)
            )

            print(
                "phi      :",
                fit[3]
            )

            print(
                "theta    :",
                fit[4]
            )

            print(
                "p        :",
                fit[9]
            )

            print(
                "q        :",
                fit[10]
            )

            print(
                "r        :",
                fit[11]
            )

        fit_free, _ = free_run(
            A,
            B,
            Xv,
            Uv
        )

        results[name][
            "free"
        ] = fit_free

        print()
        print(
            "FULL FREE-RUN mean FIT :",
            np.nanmean(fit_free)
        )

        print(
            "free p                  :",
            fit_free[9]
        )

        print(
            "free q                  :",
            fit_free[10]
        )

        print(
            "free phi                :",
            fit_free[3]
        )

        print(
            "free theta              :",
            fit_free[4]
        )

    print()
    print("=" * 112)
    print("COMPACT COMPARISON")
    print("=" * 112)

    print(
        "candidate      H1 mean    H10 mean    "
        "H20 mean    H100 mean    FULL mean    "
        "H20 p     H20 q"
    )

    for name in [
        "FULL_OLS",
        "IV_PQ",
        "IV_PQR",
    ]:

        print(
            f"{name:12s} "
            f"{np.nanmean(results[name][1]):9.3f} "
            f"{np.nanmean(results[name][10]):11.3f} "
            f"{np.nanmean(results[name][20]):11.3f} "
            f"{np.nanmean(results[name][100]):12.3f} "
            f"{np.nanmean(results[name]['free']):12.3f} "
            f"{results[name][20][9]:9.3f} "
            f"{results[name][20][10]:9.3f}"
        )

    print()
    print(
        "NOTE: pem_val_mimo_01 is DEVELOPMENT validation."
    )

    print(
        "No candidate model was saved or finalized."
    )

    print("=" * 112)


if __name__ == "__main__":
    main()
