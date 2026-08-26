#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_estimate.py

Estimate the DD21 discrete-time linear PEM model:

    dx[k+1] = A_PEM dx[k] + B_PEM du[k] + w[k]

IMPORTANT:
- Uses TRAINING data only.
- Held-out validation data is NOT loaded here.
- No intercept is estimated because each run has already
  been centered around its empirical pre-hover operating point.

Author: Abdallah GHOUL
2026
"""

import json
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

INPUT_NAMES = [
    "dT",
    "tau_phi",
    "tau_theta",
    "tau_psi",
]


def fit_percentage(y, yhat):
    """
    System-identification style fit percentage:

        FIT = 100 * (1 - ||y-yhat|| / ||y-mean(y)||)
    """

    denom = np.linalg.norm(
        y - np.mean(y)
    )

    if denom <= 1e-15:
        return np.nan

    return 100.0 * (
        1.0
        - np.linalg.norm(y - yhat) / denom
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

    processed_dir = os.path.join(
        package_dir,
        "data",
        "pem_identification",
        "processed"
    )

    train_path = os.path.join(
        processed_dir,
        "pem_train_dataset.npz"
    )

    if not os.path.isfile(train_path):
        raise RuntimeError(
            "Training dataset not found:\n{}".format(
                train_path
            )
        )

    print("=" * 78)
    print(" DD21 PEM — PARAMETER ESTIMATION")
    print("=" * 78)

    print("\nTraining dataset:")
    print(train_path)

    # ==================================================================
    # Load TRAINING only
    # ==================================================================

    data = np.load(
        train_path,
        allow_pickle=False
    )

    X = np.asarray(
        data["X_k"],
        dtype=float
    )

    U = np.asarray(
        data["U_k"],
        dtype=float
    )

    Y = np.asarray(
        data["X_kp1"],
        dtype=float
    )

    Ts = float(
        data["Ts"]
    )

    print("\nDATA")
    print("Ts     :", Ts)
    print("X_k    :", X.shape)
    print("U_k    :", U.shape)
    print("X_kp1  :", Y.shape)

    # ==================================================================
    # Sanity checks
    # ==================================================================

    if X.shape[1] != NX:
        raise RuntimeError(
            "Invalid X dimension."
        )

    if U.shape[1] != NU:
        raise RuntimeError(
            "Invalid U dimension."
        )

    if Y.shape[1] != NX:
        raise RuntimeError(
            "Invalid Y dimension."
        )

    if not (
        len(X) == len(U) == len(Y)
    ):
        raise RuntimeError(
            "Training row count mismatch."
        )

    for name, a in [
        ("X", X),
        ("U", U),
        ("Y", Y),
    ]:

        if not np.all(
            np.isfinite(a)
        ):
            raise RuntimeError(
                "{} contains NaN/Inf.".format(
                    name
                )
            )

    # ==================================================================
    # Regression matrix
    # ==================================================================

    Z = np.hstack([
        X,
        U
    ])

    rank_Z = int(
        np.linalg.matrix_rank(Z)
    )

    print("\nREGRESSION")
    print("Z shape :", Z.shape)
    print("rank    :", rank_Z)
    print("required:", NX + NU)

    if rank_Z != NX + NU:
        raise RuntimeError(
            "Regression matrix is rank deficient."
        )

    # ==================================================================
    # Ordinary Least Squares
    #
    # Y = Z @ Theta
    #
    # Theta:
    #   first 12 rows -> A.T
    #   last   4 rows -> B.T
    # ==================================================================

    Theta, residuals, rank_lstsq, singular_values = (
        np.linalg.lstsq(
            Z,
            Y,
            rcond=None
        )
    )

    A = Theta[:NX, :].T

    B = Theta[NX:, :].T

    if A.shape != (NX, NX):
        raise RuntimeError(
            "Invalid A shape."
        )

    if B.shape != (NX, NU):
        raise RuntimeError(
            "Invalid B shape."
        )

    # ==================================================================
    # Training prediction
    # ==================================================================

    Yhat = (
        X @ A.T
        + U @ B.T
    )

    E = (
        Y
        - Yhat
    )

    rmse = np.sqrt(
        np.mean(
            E ** 2,
            axis=0
        )
    )

    mae = np.mean(
        np.abs(E),
        axis=0
    )

    maxae = np.max(
        np.abs(E),
        axis=0
    )

    fit = np.asarray([
        fit_percentage(
            Y[:, i],
            Yhat[:, i]
        )
        for i in range(NX)
    ])

    # ==================================================================
    # Residual covariance
    #
    # Degrees-of-freedom corrected estimate:
    #
    # Qw = E.T E / (N - rank(Z))
    # ==================================================================

    dof = (
        len(Z)
        - rank_Z
    )

    if dof <= 0:
        raise RuntimeError(
            "Invalid residual degrees of freedom."
        )

    Qw = (
        E.T @ E
    ) / float(dof)

    # ==================================================================
    # Dynamic properties
    # ==================================================================

    eigvals = np.linalg.eigvals(
        A
    )

    spectral_radius = float(
        np.max(
            np.abs(eigvals)
        )
    )

    # ==================================================================
    # Print matrices
    # ==================================================================

    np.set_printoptions(
        precision=8,
        suppress=True,
        linewidth=180
    )

    print("\nA_PEM")
    print("-" * 78)
    print(A)

    print("\nB_PEM")
    print("-" * 78)
    print(B)

    print("\nEIGENVALUES(A_PEM)")
    print("-" * 78)
    print(eigvals)

    print(
        "\nspectral radius :",
        spectral_radius
    )

    # ==================================================================
    # Training metrics
    # ==================================================================

    print("\nTRAINING ONE-STEP METRICS")
    print("-" * 78)

    for i, name in enumerate(
        STATE_NAMES
    ):

        print(
            f"{name:8s}: "
            f"RMSE={rmse[i]:.10e}  "
            f"MAE={mae[i]:.10e}  "
            f"MaxAE={maxae[i]:.10e}  "
            f"FIT={fit[i]:8.3f}%"
        )

    print("\nRESIDUAL COVARIANCE Qw")
    print("-" * 78)
    print(Qw)

    # ==================================================================
    # Save candidate model
    #
    # It is deliberately called "candidate":
    # it must pass held-out one-step + multi-step validation
    # before becoming the final PEM model.
    # ==================================================================

    model_npz = os.path.join(
        processed_dir,
        "pem_candidate_model.npz"
    )

    np.savez_compressed(
        model_npz,

        A=A,
        B=B,
        Qw=Qw,

        Ts=np.asarray(
            Ts,
            dtype=float
        ),

        state_names=np.asarray(
            STATE_NAMES
        ),

        input_names=np.asarray(
            INPUT_NAMES
        ),

        eigenvalues=eigvals,

        spectral_radius=np.asarray(
            spectral_radius,
            dtype=float
        ),

        training_rmse=rmse,
        training_mae=mae,
        training_maxae=maxae,
        training_fit_percent=fit,

        regression_rank=np.asarray(
            rank_Z,
            dtype=int
        ),

        regression_singular_values=
            singular_values,
    )

    summary = {
        "status": (
            "candidate — requires held-out validation"
        ),

        "model": (
            "dx[k+1] = A_PEM dx[k] "
            "+ B_PEM du[k] + w[k]"
        ),

        "estimator": (
            "ordinary least squares"
        ),

        "intercept": False,

        "training_only": True,

        "Ts": Ts,

        "nx": NX,
        "nu": NU,

        "n_training_transitions":
            int(len(Z)),

        "regression_rank":
            rank_Z,

        "spectral_radius":
            spectral_radius,

        "state_names":
            STATE_NAMES,

        "input_names":
            INPUT_NAMES,

        "training_metrics": {
            STATE_NAMES[i]: {
                "rmse":
                    float(rmse[i]),
                "mae":
                    float(mae[i]),
                "maxae":
                    float(maxae[i]),
                "fit_percent":
                    float(fit[i]),
            }
            for i in range(NX)
        },

        "model_file":
            model_npz,
    }

    summary_path = os.path.join(
        processed_dir,
        "pem_candidate_model_summary.json"
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

    print("\n" + "=" * 78)
    print(" PEM CANDIDATE ESTIMATION COMPLETE")
    print("=" * 78)

    print(
        "A shape             :",
        A.shape
    )

    print(
        "B shape             :",
        B.shape
    )

    print(
        "rank(Z)             :",
        rank_Z
    )

    print(
        "spectral radius     :",
        spectral_radius
    )

    print("\nCandidate model:")
    print(model_npz)

    print("\nSummary:")
    print(summary_path)

    print()
    print(
        "[PASS] training-only estimation"
    )

    print(
        "[PASS] A_PEM estimated"
    )

    print(
        "[PASS] B_PEM estimated"
    )

    print(
        "[PASS] residual covariance estimated"
    )

    print(
        "[WAIT] held-out one-step validation"
    )

    print(
        "[WAIT] held-out multi-step validation"
    )


if __name__ == "__main__":
    main()
