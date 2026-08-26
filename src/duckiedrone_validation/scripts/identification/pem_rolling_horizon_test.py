#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_rolling_horizon_test.py

Held-out rolling-horizon validation of DD21 PEM candidate.

For every possible starting sample k:

    xhat[k|k] = x[k]

then

    xhat[k+j+1|k] =
        A xhat[k+j|k] + B u[k+j]

for j = 0 ... H-1.

The state is reset to the measured state at EVERY new
prediction origin, exactly as in receding-horizon MPC.

Validation data are never used for estimation.

Author: Abdallah GHOUL
2026
"""

import os
import numpy as np


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
]


def fit_percent(y, yh):

    denominator = np.linalg.norm(
        y - np.mean(y)
    )

    if denominator <= 1e-15:
        return np.nan

    return 100.0 * (
        1.0
        - np.linalg.norm(y - yh)
        / denominator
    )


def rolling_terminal_prediction(
        A,
        B,
        X,
        U,
        H):
    """
    Predict exactly H steps ahead from every
    possible measured starting state.

    Returns:
        true terminal states
        predicted terminal states
    """

    truth = []
    prediction = []

    n = len(X)

    # Need X[k+H] to exist.
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

        prediction.append(
            xh.copy()
        )

    return (
        np.asarray(truth),
        np.asarray(prediction)
    )


def rolling_trajectory_prediction(
        A,
        B,
        X,
        U,
        H):
    """
    Pool all prediction steps 1...H over
    every rolling prediction origin.
    """

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

            prediction.append(
                xh.copy()
            )

    return (
        np.asarray(truth),
        np.asarray(prediction)
    )


def calculate_metrics(
        Y,
        Yh):

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

    fit = np.asarray([
        fit_percent(
            Y[:, j],
            Yh[:, j]
        )
        for j in range(12)
    ])

    return (
        fit,
        rmse,
        mae
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

    model = np.load(
        os.path.join(
            root,
            "pem_candidate_model.npz"
        ),
        allow_pickle=False
    )

    validation = np.load(
        os.path.join(
            root,
            "pem_val_mimo_01",
            "processed.npz"
        ),
        allow_pickle=False
    )

    A = np.asarray(
        model["A"],
        dtype=float
    )

    B = np.asarray(
        model["B"],
        dtype=float
    )

    X = np.asarray(
        validation["X_centered"],
        dtype=float
    )

    U = np.asarray(
        validation["U_centered"],
        dtype=float
    )

    Ts = float(
        validation["Ts"]
    )

    print("=" * 100)
    print(
        "DD21 PEM — HELD-OUT ROLLING-HORIZON VALIDATION"
    )
    print("=" * 100)

    print()
    print(
        "samples :",
        len(X)
    )

    print(
        "Ts      :",
        Ts
    )

    print(
        "rho(A)  :",
        np.max(
            np.abs(
                np.linalg.eigvals(A)
            )
        )
    )

    results = {}

    for H in HORIZONS:

        # --------------------------------------------------------------
        # Terminal H-step prediction
        # --------------------------------------------------------------

        Yt, Yh = (
            rolling_terminal_prediction(
                A,
                B,
                X,
                U,
                H
            )
        )

        (
            terminal_fit,
            terminal_rmse,
            terminal_mae
        ) = calculate_metrics(
            Yt,
            Yh
        )

        # --------------------------------------------------------------
        # Entire trajectory 1 ... H
        # --------------------------------------------------------------

        Ytraj, Yhtraj = (
            rolling_trajectory_prediction(
                A,
                B,
                X,
                U,
                H
            )
        )

        (
            traj_fit,
            traj_rmse,
            traj_mae
        ) = calculate_metrics(
            Ytraj,
            Yhtraj
        )

        results[H] = {
            "terminal_fit":
                terminal_fit,
            "trajectory_fit":
                traj_fit,
        }

        print()
        print("-" * 100)

        print(
            f"H = {H:2d} steps "
            f"({H * Ts:.2f} s)"
        )

        print()
        print(
            "ROLLING TERMINAL H-STEP"
        )

        print(
            "mean FIT :",
            np.nanmean(
                terminal_fit
            )
        )

        print()
        print(
            "state        FIT [%]          RMSE"
        )

        for j, name in enumerate(
            STATE_NAMES
        ):

            print(
                f"{name:8s}"
                f" {terminal_fit[j]:12.3f}"
                f" {terminal_rmse[j]:14.6e}"
            )

        print()
        print(
            "ROLLING TRAJECTORY 1...H"
        )

        print(
            "mean FIT :",
            np.nanmean(
                traj_fit
            )
        )

        print(
            "p FIT    :",
            traj_fit[9]
        )

        print(
            "q FIT    :",
            traj_fit[10]
        )

        print(
            "phi FIT  :",
            traj_fit[3]
        )

        print(
            "theta FIT:",
            traj_fit[4]
        )

    print()
    print("=" * 100)
    print(
        "MPC-HORIZON SUMMARY"
    )
    print("=" * 100)

    print(
        "H     time[s]    terminal mean FIT    "
        "trajectory mean FIT"
    )

    for H in HORIZONS:

        print(
            f"{H:2d}"
            f"    {H*Ts:7.2f}"
            f"       "
            f"{np.nanmean(results[H]['terminal_fit']):10.3f}%"
            f"             "
            f"{np.nanmean(results[H]['trajectory_fit']):10.3f}%"
        )

    print()
    print(
        "Current MPC prediction horizon:"
    )

    print(
        "Np = 20 ->",
        20 * Ts,
        "s"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
