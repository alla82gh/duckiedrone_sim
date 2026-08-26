#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_training_composition_sweep.py

Diagnostic comparison of PEM training compositions:

1. SISO_ONLY
2. MIMO_ONLY
3. ALL = SISO + MIMO

All models are estimated by identical OLS formulation:

    x[k+1] = A x[k] + B u[k]

Validation is always performed on the same completely held-out:

    pem_val_mimo_01

No validation data are used during estimation.

Author: Abdallah GHOUL
2026
"""

import os
import numpy as np


NX = 12
NU = 4

STATE_NAMES = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]

SISO_RUNS = [
    "pem_train_thrust_01",
    "pem_train_roll_01",
    "pem_train_pitch_01",
    "pem_train_yaw_01",
]

MIMO_RUNS = [
    "pem_train_mimo_01",
]

CONFIGS = {
    "SISO_ONLY": SISO_RUNS,
    "MIMO_ONLY": MIMO_RUNS,
    "ALL": SISO_RUNS + MIMO_RUNS,
}

VAL_RUN = "pem_val_mimo_01"

HORIZONS = [
    1,
    20,
    100,
    1497,
]


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


def load_run(root, run_name):

    path = os.path.join(
        root,
        run_name,
        "processed.npz"
    )

    if not os.path.isfile(path):
        raise RuntimeError(
            "Missing processed run:\n{}".format(
                path
            )
        )

    d = np.load(
        path,
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

    return X, U


def transitions(X, U):

    if len(X) != len(U):
        raise RuntimeError(
            "X/U length mismatch."
        )

    return (
        X[:-1, :],
        U[:-1, :],
        X[1:, :]
    )


def build_training(root, run_names):

    xs = []
    us = []
    ys = []

    for run_name in run_names:

        X, U = load_run(
            root,
            run_name
        )

        Xk, Uk, Y = transitions(
            X,
            U
        )

        xs.append(Xk)
        us.append(Uk)
        ys.append(Y)

    return (
        np.vstack(xs),
        np.vstack(us),
        np.vstack(ys)
    )


def estimate(X, U, Y):

    Z = np.hstack([
        X,
        U
    ])

    rank = int(
        np.linalg.matrix_rank(Z)
    )

    std = np.std(
        Z,
        axis=0
    )

    Zs = (
        Z
        - np.mean(
            Z,
            axis=0,
            keepdims=True
        )
    ) / std

    svals = np.linalg.svd(
        Zs,
        compute_uv=False
    )

    cond = float(
        svals[0] / svals[-1]
    )

    Theta, _, _, _ = np.linalg.lstsq(
        Z,
        Y,
        rcond=None
    )

    A = Theta[:NX, :].T
    B = Theta[NX:, :].T

    return (
        A,
        B,
        rank,
        cond
    )


def horizon_prediction(
        A,
        B,
        X,
        U,
        H):

    predictions = []
    truth = []

    N = len(X) - 1

    start = 0

    while start < N:

        stop = min(
            start + H,
            N
        )

        xh = X[start, :].copy()

        for k in range(
            start,
            stop
        ):

            xh = (
                A @ xh
                + B @ U[k, :]
            )

            predictions.append(
                xh.copy()
            )

            truth.append(
                X[k + 1, :].copy()
            )

        start = stop

    Y = np.asarray(
        truth
    )

    Yh = np.asarray(
        predictions
    )

    fits = np.asarray([
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

    return fits, rmse


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

    Xv, Uv = load_run(
        root,
        VAL_RUN
    )

    print("=" * 100)
    print("DD21 PEM — TRAINING COMPOSITION SWEEP")
    print("=" * 100)

    print()
    print(
        "Held-out validation:",
        VAL_RUN
    )

    results = {}

    for config_name, runs in CONFIGS.items():

        print()
        print("=" * 100)
        print(config_name)
        print("=" * 100)

        for r in runs:
            print("training run:", r)

        X, U, Y = build_training(
            root,
            runs
        )

        (
            A,
            B,
            rank,
            cond
        ) = estimate(
            X,
            U,
            Y
        )

        eig = np.linalg.eigvals(
            A
        )

        rho = float(
            np.max(
                np.abs(eig)
            )
        )

        print()
        print("TRAINING MATRIX")
        print("transitions :", len(X))
        print("rank        :", rank)
        print("required    :", NX + NU)
        print(
            "std cond    :",
            cond
        )

        print()
        print("MODEL")
        print(
            "spectral radius :",
            rho
        )

        print()
        print("KEY COEFFICIENTS")

        print(
            "A[x,vx]        =",
            A[0, 6]
        )

        print(
            "A[y,vy]        =",
            A[1, 7]
        )

        print(
            "A[z,vz]        =",
            A[2, 8]
        )

        print(
            "A[vx,theta]    =",
            A[6, 4]
        )

        print(
            "A[vy,phi]      =",
            A[7, 3]
        )

        print(
            "B[vz,dT]       =",
            B[8, 0]
        )

        print(
            "B[p,tau_phi]   =",
            B[9, 1]
        )

        print(
            "B[q,tau_theta] =",
            B[10, 2]
        )

        print(
            "B[r,tau_psi]   =",
            B[11, 3]
        )

        print(
            "A[p,phi]       =",
            A[9, 3]
        )

        print(
            "A[q,theta]     =",
            A[10, 4]
        )

        config_results = {
            "A": A,
            "B": B,
            "rank": rank,
            "cond": cond,
            "rho": rho,
            "horizons": {},
        }

        for H in HORIZONS:

            fits, rmse = horizon_prediction(
                A,
                B,
                Xv,
                Uv,
                H
            )

            config_results[
                "horizons"
            ][H] = fits

            print()
            print(
                f"H = {H:4d} "
                f"({H * 0.01:.2f} s)"
            )

            print(
                "mean FIT :",
                np.nanmean(
                    fits
                )
            )

            print(
                "p FIT    :",
                fits[9]
            )

            print(
                "q FIT    :",
                fits[10]
            )

            print(
                "phi FIT  :",
                fits[3]
            )

            print(
                "theta FIT:",
                fits[4]
            )

        results[
            config_name
        ] = config_results

    # ------------------------------------------------------------------
    # Compact comparison
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("COMPACT COMPARISON")
    print("=" * 100)

    print(
        "config       Ntrain   rank   cond     rho(A)"
        "      H1 FIT    H20 FIT   H100 FIT   FULL FIT"
    )

    for name in [
        "SISO_ONLY",
        "MIMO_ONLY",
        "ALL",
    ]:

        r = results[name]

        ntrain = sum(
            len(
                transitions(
                    *load_run(
                        root,
                        run
                    )
                )[0]
            )
            for run in CONFIGS[name]
        )

        h1 = np.nanmean(
            r["horizons"][1]
        )

        h20 = np.nanmean(
            r["horizons"][20]
        )

        h100 = np.nanmean(
            r["horizons"][100]
        )

        hfull = np.nanmean(
            r["horizons"][1497]
        )

        print(
            f"{name:11s} "
            f"{ntrain:7d} "
            f"{r['rank']:6d} "
            f"{r['cond']:7.3f} "
            f"{r['rho']:10.6f} "
            f"{h1:10.3f} "
            f"{h20:10.3f} "
            f"{h100:11.3f} "
            f"{hfull:10.3f}"
        )

    print()
    print(
        "Expected physical references:"
    )

    print(
        "A[vx,theta]    ~",
        9.81 * 0.01
    )

    print(
        "A[vy,phi]      ~",
        -9.81 * 0.01
    )

    print(
        "B[vz,dT]       ~",
        0.01 / 0.635
    )

    print(
        "B[p,tau_phi]   ~",
        0.01 / 0.0015
    )

    print(
        "B[q,tau_theta] ~",
        0.01 / 0.0017
    )

    print(
        "B[r,tau_psi]   ~",
        0.01 / 0.0030
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
