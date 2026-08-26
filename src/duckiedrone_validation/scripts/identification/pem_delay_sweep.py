#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_delay_sweep.py

Empirical input-delay diagnosis for DD21 PEM identification.

Tests:

    x[k+1] = A x[k] + B u[k-d]

for:

    d = 0, 1, 2, 3 samples

IMPORTANT:
- A,B are estimated from TRAINING runs only.
- pem_val_mimo_01 is held out.
- Transitions never cross run boundaries.
- No delay is accepted a priori.

Author: Abdallah GHOUL
2026
"""

import os
import numpy as np


TRAIN_RUNS = [
    "pem_train_thrust_01",
    "pem_train_roll_01",
    "pem_train_pitch_01",
    "pem_train_yaw_01",
    "pem_train_mimo_01",
]

VAL_RUN = "pem_val_mimo_01"

STATE_NAMES = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]

NX = 12
NU = 4

DELAYS = [0, 1, 2, 3]

HORIZON = 20


def fit_percent(y, yh):

    denom = np.linalg.norm(
        y - np.mean(y)
    )

    if denom <= 1e-15:
        return np.nan

    return 100.0 * (
        1.0
        - np.linalg.norm(y - yh) / denom
    )


def load_run(root, run_name):

    path = os.path.join(
        root,
        run_name,
        "processed.npz"
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


def delayed_transitions(X, U, delay):
    """
    Build:

        x[k+1] = A x[k] + B u[k-delay]

    inside one run only.
    """

    n = len(X)

    if n <= delay + 1:
        raise RuntimeError(
            "Run too short for delay."
        )

    k = np.arange(
        delay,
        n - 1
    )

    Xk = X[k, :]
    Uk = U[k - delay, :]
    Y = X[k + 1, :]

    return Xk, Uk, Y


def estimate_model(root, delay):

    X_all = []
    U_all = []
    Y_all = []

    for run_name in TRAIN_RUNS:

        X, U = load_run(
            root,
            run_name
        )

        Xk, Uk, Y = delayed_transitions(
            X,
            U,
            delay
        )

        X_all.append(Xk)
        U_all.append(Uk)
        Y_all.append(Y)

    Xtr = np.vstack(X_all)
    Utr = np.vstack(U_all)
    Ytr = np.vstack(Y_all)

    Z = np.hstack([
        Xtr,
        Utr
    ])

    Theta, _, _, _ = np.linalg.lstsq(
        Z,
        Ytr,
        rcond=None
    )

    A = Theta[:NX, :].T
    B = Theta[NX:, :].T

    return A, B, Xtr, Utr, Ytr


def one_step_metrics(A, B, X, U, delay):

    Xk, Uk, Y = delayed_transitions(
        X,
        U,
        delay
    )

    Yh = (
        Xk @ A.T
        + Uk @ B.T
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


def horizon_metrics(
        A,
        B,
        X,
        U,
        delay,
        H):
    """
    H-step block prediction with measured reset
    every H transitions.

    The delayed input history is taken from the
    recorded run.
    """

    predictions = []
    truth = []

    n = len(X)

    start = delay

    last_transition = n - 1

    while start < last_transition:

        stop = min(
            start + H,
            last_transition
        )

        xh = X[start, :].copy()

        for k in range(
            start,
            stop
        ):

            u_delayed = U[
                k - delay,
                :
            ]

            xh = (
                A @ xh
                + B @ u_delayed
            )

            predictions.append(
                xh.copy()
            )

            truth.append(
                X[k + 1, :].copy()
            )

        start = stop

    Y = np.asarray(truth)
    Yh = np.asarray(predictions)

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

    print("=" * 96)
    print("DD21 PEM — INPUT DELAY SWEEP")
    print("=" * 96)

    print()
    print(
        "Model: x[k+1] = A x[k] + B u[k-d]"
    )

    print(
        "Validation horizon:",
        HORIZON,
        "steps =",
        HORIZON * 0.01,
        "s"
    )

    results = []

    for delay in DELAYS:

        (
            A,
            B,
            Xtr,
            Utr,
            Ytr
        ) = estimate_model(
            root,
            delay
        )

        eig = np.linalg.eigvals(A)

        rho = float(
            np.max(
                np.abs(eig)
            )
        )

        one_fit, one_rmse = (
            one_step_metrics(
                A,
                B,
                Xv,
                Uv,
                delay
            )
        )

        h_fit, h_rmse = (
            horizon_metrics(
                A,
                B,
                Xv,
                Uv,
                delay,
                HORIZON
            )
        )

        result = {
            "delay": delay,
            "A": A,
            "B": B,
            "rho": rho,
            "one_fit": one_fit,
            "h_fit": h_fit,
        }

        results.append(result)

        print()
        print("-" * 96)

        print(
            f"DELAY d = {delay} samples "
            f"({delay * 0.01:.2f} s)"
        )

        print(
            "training transitions :",
            len(Xtr)
        )

        print(
            "spectral radius      :",
            rho
        )

        print(
            "held-out 1-step FIT  :",
            np.nanmean(one_fit)
        )

        print(
            "held-out H=20 FIT    :",
            np.nanmean(h_fit)
        )

        print()
        print("KEY DYNAMIC CHANNELS")

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

        print()
        print("H=20 STATE FIT")

        for j, name in enumerate(
            STATE_NAMES
        ):

            print(
                f"{name:8s}: "
                f"{h_fit[j]:8.3f}%"
            )

    print()
    print("=" * 96)
    print("COMPACT COMPARISON")
    print("=" * 96)

    print(
        "delay   rho(A)       "
        "1-step mean FIT    "
        "H20 mean FIT      "
        "H20 p      H20 q"
    )

    for r in results:

        print(
            f"{r['delay']:5d}   "
            f"{r['rho']:10.6f}   "
            f"{np.nanmean(r['one_fit']):15.3f}   "
            f"{np.nanmean(r['h_fit']):12.3f}   "
            f"{r['h_fit'][9]:8.3f}   "
            f"{r['h_fit'][10]:8.3f}"
        )

    best = max(
        results,
        key=lambda r:
            np.nanmean(
                r["h_fit"]
            )
    )

    print()
    print(
        "Best H20 delay candidate:",
        best["delay"],
        "samples"
    )

    print(
        "Best H20 mean FIT:",
        np.nanmean(
            best["h_fit"]
        )
    )

    print()
    print(
        "NOTE: this is a timing diagnosis only."
    )

    print(
        "No delay is accepted until the result "
        "is interpreted physically."
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
