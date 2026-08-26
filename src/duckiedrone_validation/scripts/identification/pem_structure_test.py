#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_structure_test.py

Diagnostic comparison:

1. FULL unconstrained state-space OLS model
2. STRUCTURED sparse state-space model

Both models use EXACTLY the same frozen ALL training dataset.

Validation uses only:

    pem_val_mimo_01

No validation data are used for parameter estimation.

Structured model near hover:

    x+     <- x, vx
    y+     <- y, vy
    z+     <- z, vz

    phi+   <- phi, p
    theta+ <- theta, q
    psi+   <- psi, r

    vx+    <- vx, theta
    vy+    <- vy, phi
    vz+    <- vz, dT

    p+     <- p, tau_phi
    q+     <- q, tau_theta
    r+     <- r, tau_psi

The numerical coefficients remain entirely data-estimated.

Author: Abdallah GHOUL
2026
"""

import os
import numpy as np


NX = 12
NU = 4
TS = 0.01

STATE_NAMES = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]

INPUT_NAMES = [
    "dT",
    "tau_phi",
    "tau_theta",
    "tau_psi",
]


# ----------------------------------------------------------------------
# Allowed regressors for each state equation.
#
# Entries:
#   ("x", index) -> state regressor
#   ("u", index) -> input regressor
# ----------------------------------------------------------------------

STRUCTURE = {

    # Position kinematics
    0: [("x", 0), ("x", 6)],
    1: [("x", 1), ("x", 7)],
    2: [("x", 2), ("x", 8)],

    # Attitude kinematics
    3: [("x", 3), ("x", 9)],
    4: [("x", 4), ("x", 10)],
    5: [("x", 5), ("x", 11)],

    # Translational dynamics
    6: [("x", 6), ("x", 4)],
    7: [("x", 7), ("x", 3)],
    8: [("x", 8), ("u", 0)],

    # Angular-rate dynamics
    9:  [("x", 9),  ("u", 1)],
    10: [("x", 10), ("u", 2)],
    11: [("x", 11), ("u", 3)],
}


def fit_percent(y, yh):

    denom = np.linalg.norm(
        y - np.mean(y)
    )

    if denom <= 1.0e-15:
        return np.nan

    return 100.0 * (
        1.0
        - np.linalg.norm(y - yh) / denom
    )


def metrics(Y, Yh):

    rmse = np.sqrt(
        np.mean(
            (Y - Yh) ** 2,
            axis=0
        )
    )

    fit = np.asarray([
        fit_percent(
            Y[:, j],
            Yh[:, j]
        )
        for j in range(NX)
    ])

    return fit, rmse


def estimate_full(X, U, Y):

    Z = np.hstack([
        X,
        U
    ])

    Theta, _, _, _ = np.linalg.lstsq(
        Z,
        Y,
        rcond=None
    )

    A = Theta[:NX, :].T
    B = Theta[NX:, :].T

    return A, B


def estimate_structured(X, U, Y):

    A = np.zeros(
        (NX, NX),
        dtype=float
    )

    B = np.zeros(
        (NX, NU),
        dtype=float
    )

    for row in range(NX):

        spec = STRUCTURE[row]

        columns = []

        for source, index in spec:

            if source == "x":
                columns.append(
                    X[:, index]
                )

            elif source == "u":
                columns.append(
                    U[:, index]
                )

            else:
                raise RuntimeError(
                    "Unknown regressor source."
                )

        Zrow = np.column_stack(
            columns
        )

        theta, _, _, _ = np.linalg.lstsq(
            Zrow,
            Y[:, row],
            rcond=None
        )

        for coefficient, (
                source,
                index) in zip(
                    theta,
                    spec):

            if source == "x":

                A[
                    row,
                    index
                ] = coefficient

            else:

                B[
                    row,
                    index
                ] = coefficient

    return A, B


def one_step(
        A,
        B,
        X,
        U,
        Y):

    Yh = (
        X @ A.T
        + U @ B.T
    )

    return metrics(
        Y,
        Yh
    )


def horizon_prediction(
        A,
        B,
        X,
        U,
        H):
    """
    Block H-step prediction.

    Model resets to measured state only
    at the beginning of each H-step block.
    """

    predictions = []
    truth = []

    n_transitions = (
        len(X) - 1
    )

    start = 0

    while start < n_transitions:

        stop = min(
            start + H,
            n_transitions
        )

        xh = X[
            start,
            :
        ].copy()

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

    return metrics(
        Y,
        Yh
    )


def print_model_summary(
        title,
        A,
        B):

    eig = np.linalg.eigvals(
        A
    )

    rho = float(
        np.max(
            np.abs(eig)
        )
    )

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)

    print(
        "spectral radius :",
        rho
    )

    print()
    print("KINEMATIC COEFFICIENTS")

    print(
        "A[x,x]         =",
        A[0, 0]
    )

    print(
        "A[x,vx]        =",
        A[0, 6],
        " expected ~",
        TS
    )

    print(
        "A[y,y]         =",
        A[1, 1]
    )

    print(
        "A[y,vy]        =",
        A[1, 7],
        " expected ~",
        TS
    )

    print(
        "A[z,z]         =",
        A[2, 2]
    )

    print(
        "A[z,vz]        =",
        A[2, 8],
        " expected ~",
        TS
    )

    print(
        "A[phi,phi]     =",
        A[3, 3]
    )

    print(
        "A[phi,p]       =",
        A[3, 9],
        " expected ~",
        TS
    )

    print(
        "A[theta,theta] =",
        A[4, 4]
    )

    print(
        "A[theta,q]     =",
        A[4, 10],
        " expected ~",
        TS
    )

    print(
        "A[psi,psi]     =",
        A[5, 5]
    )

    print(
        "A[psi,r]       =",
        A[5, 11],
        " expected ~",
        TS
    )

    print()
    print("TRANSLATIONAL DYNAMICS")

    print(
        "A[vx,vx]       =",
        A[6, 6]
    )

    print(
        "A[vx,theta]    =",
        A[6, 4],
        " expected ~",
        9.81 * TS
    )

    print(
        "A[vy,vy]       =",
        A[7, 7]
    )

    print(
        "A[vy,phi]      =",
        A[7, 3],
        " expected ~",
        -9.81 * TS
    )

    print(
        "A[vz,vz]       =",
        A[8, 8]
    )

    print(
        "B[vz,dT]       =",
        B[8, 0],
        " expected ~",
        TS / 0.635
    )

    print()
    print("ANGULAR-RATE DYNAMICS")

    print(
        "A[p,p]         =",
        A[9, 9]
    )

    print(
        "B[p,tau_phi]   =",
        B[9, 1],
        " expected ~",
        TS / 0.0015
    )

    print(
        "A[q,q]         =",
        A[10, 10]
    )

    print(
        "B[q,tau_theta] =",
        B[10, 2],
        " expected ~",
        TS / 0.0017
    )

    print(
        "A[r,r]         =",
        A[11, 11]
    )

    print(
        "B[r,tau_psi]   =",
        B[11, 3],
        " expected ~",
        TS / 0.0030
    )

    return rho


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

    train_path = os.path.join(
        root,
        "pem_train_dataset.npz"
    )

    val_run_path = os.path.join(
        root,
        "pem_val_mimo_01",
        "processed.npz"
    )

    train = np.load(
        train_path,
        allow_pickle=False
    )

    val = np.load(
        val_run_path,
        allow_pickle=False
    )

    Xtr = np.asarray(
        train["X_k"],
        dtype=float
    )

    Utr = np.asarray(
        train["U_k"],
        dtype=float
    )

    Ytr = np.asarray(
        train["X_kp1"],
        dtype=float
    )

    Xv = np.asarray(
        val["X_centered"],
        dtype=float
    )

    Uv = np.asarray(
        val["U_centered"],
        dtype=float
    )

    print("=" * 96)
    print("DD21 PEM — STRUCTURE DIAGNOSTIC")
    print("=" * 96)

    print()
    print("TRAINING")
    print(
        "X :",
        Xtr.shape
    )

    print(
        "U :",
        Utr.shape
    )

    print(
        "Y :",
        Ytr.shape
    )

    print()
    print("VALIDATION")
    print(
        "X :",
        Xv.shape
    )

    print(
        "U :",
        Uv.shape
    )

    # ------------------------------------------------------------------
    # Estimate both models from identical training data.
    # ------------------------------------------------------------------

    A_full, B_full = estimate_full(
        Xtr,
        Utr,
        Ytr
    )

    A_struct, B_struct = (
        estimate_structured(
            Xtr,
            Utr,
            Ytr
        )
    )

    rho_full = print_model_summary(
        "FULL UNCONSTRAINED MODEL",
        A_full,
        B_full
    )

    rho_struct = print_model_summary(
        "STRUCTURED MODEL",
        A_struct,
        B_struct
    )

    # ------------------------------------------------------------------
    # Training one-step comparison
    # ------------------------------------------------------------------

    full_train_fit, _ = one_step(
        A_full,
        B_full,
        Xtr,
        Utr,
        Ytr
    )

    struct_train_fit, _ = one_step(
        A_struct,
        B_struct,
        Xtr,
        Utr,
        Ytr
    )

    print()
    print("=" * 96)
    print("TRAINING ONE-STEP")
    print("=" * 96)

    print(
        "FULL mean FIT       :",
        np.nanmean(
            full_train_fit
        )
    )

    print(
        "STRUCTURED mean FIT :",
        np.nanmean(
            struct_train_fit
        )
    )

    # ------------------------------------------------------------------
    # Held-out validation
    # ------------------------------------------------------------------

    horizons = [
        1,
        20,
        100,
        len(Xv) - 1,
    ]

    results = {}

    for label, A, B in [
        (
            "FULL",
            A_full,
            B_full
        ),
        (
            "STRUCTURED",
            A_struct,
            B_struct
        ),
    ]:

        results[label] = {}

        print()
        print("=" * 96)
        print(
            "HELD-OUT VALIDATION —",
            label
        )
        print("=" * 96)

        for H in horizons:

            fit, rmse = horizon_prediction(
                A,
                B,
                Xv,
                Uv,
                H
            )

            results[
                label
            ][H] = {
                "fit": fit,
                "rmse": rmse,
            }

            print()
            print(
                f"H = {H:4d} "
                f"({H * TS:.2f} s)"
            )

            print(
                "mean FIT :",
                np.nanmean(
                    fit
                )
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
                "vx       :",
                fit[6]
            )

            print(
                "vy       :",
                fit[7]
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

    # ------------------------------------------------------------------
    # Compact comparison
    # ------------------------------------------------------------------

    print()
    print("=" * 110)
    print("COMPACT COMPARISON")
    print("=" * 110)

    print(
        "model          rho(A)"
        "       H1 mean"
        "      H20 mean"
        "     H100 mean"
        "      FULL mean"
        "       H20 p"
        "       H20 q"
    )

    for label, rho in [
        ("FULL", rho_full),
        ("STRUCTURED", rho_struct),
    ]:

        H1 = horizons[0]
        H20 = 20
        H100 = 100
        HFULL = horizons[-1]

        print(
            f"{label:12s} "
            f"{rho:10.6f} "
            f"{np.nanmean(results[label][H1]['fit']):12.3f} "
            f"{np.nanmean(results[label][H20]['fit']):13.3f} "
            f"{np.nanmean(results[label][H100]['fit']):13.3f} "
            f"{np.nanmean(results[label][HFULL]['fit']):14.3f} "
            f"{results[label][H20]['fit'][9]:11.3f} "
            f"{results[label][H20]['fit'][10]:11.3f}"
        )

    print()
    print(
        "NOTE:"
    )

    print(
        "Structured coefficients are still "
        "estimated from data."
    )

    print(
        "Only the allowed near-hover sparsity "
        "pattern is imposed."
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
