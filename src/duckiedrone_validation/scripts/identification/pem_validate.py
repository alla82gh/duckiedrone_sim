#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_validate.py

Held-out validation of the DD21 PEM candidate model.

The candidate model was estimated using TRAINING DATA ONLY:

    dx[k+1] = A_PEM dx[k] + B_PEM du[k]

This script uses ONLY the held-out validation dataset.

Two validation modes are performed:

1. One-step prediction

       xhat[k+1] = A x[k] + B u[k]

   where the measured x[k] is supplied at every step.

2. Free-run / multi-step simulation

       xhat[k+1] = A xhat[k] + B u[k]

   where only the first measured state of each validation run
   initializes the model.

IMPORTANT:
- Validation data are never used for estimation.
- Each validation run is simulated independently.
- No transition is created across run boundaries.

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


def fit_percentage(y, yhat):
    """
    System-identification FIT percentage.

        FIT = 100 * (
            1 - ||y-yhat|| / ||y-mean(y)||
        )
    """

    y = np.asarray(
        y,
        dtype=float
    )

    yhat = np.asarray(
        yhat,
        dtype=float
    )

    denom = np.linalg.norm(
        y - np.mean(y)
    )

    if denom <= 1.0e-15:
        return np.nan

    return 100.0 * (
        1.0
        - np.linalg.norm(
            y - yhat
        ) / denom
    )


def compute_metrics(y, yhat):
    """
    Compute state-wise validation metrics.
    """

    e = (
        y
        - yhat
    )

    rmse = np.sqrt(
        np.mean(
            e ** 2,
            axis=0
        )
    )

    mae = np.mean(
        np.abs(e),
        axis=0
    )

    maxae = np.max(
        np.abs(e),
        axis=0
    )

    fit = np.asarray([
        fit_percentage(
            y[:, i],
            yhat[:, i]
        )
        for i in range(NX)
    ])

    return {
        "error": e,
        "rmse": rmse,
        "mae": mae,
        "maxae": maxae,
        "fit": fit,
    }


def print_metrics(title, metrics):
    """
    Print state-wise metrics.
    """

    print()
    print(title)
    print("-" * 78)

    for i, name in enumerate(
        STATE_NAMES
    ):

        print(
            f"{name:8s}: "
            f"RMSE={metrics['rmse'][i]:.10e}  "
            f"MAE={metrics['mae'][i]:.10e}  "
            f"MaxAE={metrics['maxae'][i]:.10e}  "
            f"FIT={metrics['fit'][i]:8.3f}%"
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

    model_path = os.path.join(
        processed_dir,
        "pem_candidate_model.npz"
    )

    val_path = os.path.join(
        processed_dir,
        "pem_val_dataset.npz"
    )

    if not os.path.isfile(
        model_path
    ):
        raise RuntimeError(
            "Candidate model not found:\n{}".format(
                model_path
            )
        )

    if not os.path.isfile(
        val_path
    ):
        raise RuntimeError(
            "Validation dataset not found:\n{}".format(
                val_path
            )
        )

    print("=" * 78)
    print(" DD21 PEM — HELD-OUT VALIDATION")
    print("=" * 78)

    print()
    print("Candidate model:")
    print(model_path)

    print()
    print("Held-out dataset:")
    print(val_path)

    # ==================================================================
    # Load candidate model
    # ==================================================================

    model = np.load(
        model_path,
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

    Ts_model = float(
        model["Ts"]
    )

    spectral_radius = float(
        model["spectral_radius"]
    )

    if A.shape != (NX, NX):
        raise RuntimeError(
            "Invalid A shape."
        )

    if B.shape != (NX, NU):
        raise RuntimeError(
            "Invalid B shape."
        )

    # ==================================================================
    # Load HELD-OUT validation data only
    # ==================================================================

    val = np.load(
        val_path,
        allow_pickle=False
    )

    X = np.asarray(
        val["X_k"],
        dtype=float
    )

    U = np.asarray(
        val["U_k"],
        dtype=float
    )

    Y = np.asarray(
        val["X_kp1"],
        dtype=float
    )

    Ts_val = float(
        val["Ts"]
    )

    segment_lengths = np.asarray(
        val["segment_lengths"],
        dtype=int
    )

    run_names = np.asarray(
        val["run_names"]
    )

    print()
    print("MODEL")
    print("-" * 78)

    print(
        "A shape         :",
        A.shape
    )

    print(
        "B shape         :",
        B.shape
    )

    print(
        "Ts              :",
        Ts_model
    )

    print(
        "spectral radius :",
        spectral_radius
    )

    print()
    print("VALIDATION DATA")
    print("-" * 78)

    print(
        "X_k             :",
        X.shape
    )

    print(
        "U_k             :",
        U.shape
    )

    print(
        "X_kp1           :",
        Y.shape
    )

    print(
        "run_names       :",
        run_names
    )

    print(
        "segment_lengths :",
        segment_lengths
    )

    # ==================================================================
    # Sanity checks
    # ==================================================================

    if not np.isclose(
        Ts_model,
        Ts_val,
        atol=1.0e-12,
        rtol=0.0
    ):
        raise RuntimeError(
            "Model/data sampling times differ."
        )

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
        len(X)
        == len(U)
        == len(Y)
    ):
        raise RuntimeError(
            "Validation row count mismatch."
        )

    if int(
        np.sum(
            segment_lengths
        )
    ) != len(X):

        raise RuntimeError(
            "Validation segment lengths do not "
            "sum to dataset length."
        )

    for name, a in [
        ("A", A),
        ("B", B),
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
    # 1. HELD-OUT ONE-STEP PREDICTION
    # ==================================================================

    Yhat_one = (
        X @ A.T
        + U @ B.T
    )

    one_step = compute_metrics(
        Y,
        Yhat_one
    )

    print_metrics(
        "HELD-OUT ONE-STEP METRICS",
        one_step
    )

    # ==================================================================
    # 2. HELD-OUT FREE-RUN / MULTI-STEP
    #
    # Each validation run starts from its own measured first state.
    # No resetting occurs inside the run.
    # ==================================================================

    Yhat_free = np.empty_like(
        Y
    )

    segment_results = []

    offset = 0

    for seg_idx, seg_len in enumerate(
        segment_lengths
    ):

        seg_len = int(
            seg_len
        )

        start = offset
        stop = (
            offset
            + seg_len
        )

        X_seg = X[
            start:stop,
            :
        ]

        U_seg = U[
            start:stop,
            :
        ]

        Y_seg = Y[
            start:stop,
            :
        ]

        if seg_len <= 0:
            raise RuntimeError(
                "Empty validation segment."
            )

        # Initial measured state only.
        xhat = np.array(
            X_seg[0, :],
            dtype=float,
            copy=True
        )

        Yhat_seg = np.empty_like(
            Y_seg
        )

        for k in range(
            seg_len
        ):

            xhat = (
                A @ xhat
                + B @ U_seg[k, :]
            )

            Yhat_seg[
                k,
                :
            ] = xhat

        Yhat_free[
            start:stop,
            :
        ] = Yhat_seg

        seg_metrics = compute_metrics(
            Y_seg,
            Yhat_seg
        )

        run_name = str(
            run_names[seg_idx]
        )

        segment_results.append({
            "run_name":
                run_name,

            "n_transitions":
                seg_len,

            "rmse":
                seg_metrics["rmse"],

            "mae":
                seg_metrics["mae"],

            "maxae":
                seg_metrics["maxae"],

            "fit":
                seg_metrics["fit"],
        })

        offset = stop

    free_run = compute_metrics(
        Y,
        Yhat_free
    )

    print_metrics(
        "HELD-OUT FREE-RUN / MULTI-STEP METRICS",
        free_run
    )

    # ==================================================================
    # Segment metrics
    # ==================================================================

    print()
    print("FREE-RUN SEGMENT SUMMARY")
    print("-" * 78)

    for result in segment_results:

        print(
            "run        :",
            result["run_name"]
        )

        print(
            "transitions:",
            result["n_transitions"]
        )

        print(
            "mean FIT   :",
            np.nanmean(
                result["fit"]
            )
        )

        print()

    # ==================================================================
    # Overall compact summary
    # ==================================================================

    print("=" * 78)
    print(" VALIDATION SUMMARY")
    print("=" * 78)

    print(
        "one-step mean FIT :",
        np.nanmean(
            one_step["fit"]
        )
    )

    print(
        "free-run mean FIT :",
        np.nanmean(
            free_run["fit"]
        )
    )

    print(
        "one-step max RMSE :",
        np.max(
            one_step["rmse"]
        )
    )

    print(
        "free-run max RMSE :",
        np.max(
            free_run["rmse"]
        )
    )

    print(
        "spectral radius   :",
        spectral_radius
    )

    # ==================================================================
    # Save validation results
    # ==================================================================

    result_npz = os.path.join(
        processed_dir,
        "pem_validation_results.npz"
    )

    np.savez_compressed(
        result_npz,

        A=A,
        B=B,

        Ts=np.asarray(
            Ts_model,
            dtype=float
        ),

        X_k=X,
        U_k=U,
        X_kp1=Y,

        Yhat_one_step=
            Yhat_one,

        Yhat_free_run=
            Yhat_free,

        one_step_error=
            one_step["error"],

        free_run_error=
            free_run["error"],

        one_step_rmse=
            one_step["rmse"],

        one_step_mae=
            one_step["mae"],

        one_step_maxae=
            one_step["maxae"],

        one_step_fit_percent=
            one_step["fit"],

        free_run_rmse=
            free_run["rmse"],

        free_run_mae=
            free_run["mae"],

        free_run_maxae=
            free_run["maxae"],

        free_run_fit_percent=
            free_run["fit"],
    )

    summary = {
        "status":
            "held-out validation completed",

        "validation_data_used_for_estimation":
            False,

        "Ts":
            Ts_model,

        "spectral_radius":
            spectral_radius,

        "n_validation_transitions":
            int(len(X)),

        "one_step_mean_fit_percent":
            float(
                np.nanmean(
                    one_step["fit"]
                )
            ),

        "free_run_mean_fit_percent":
            float(
                np.nanmean(
                    free_run["fit"]
                )
            ),

        "one_step": {
            STATE_NAMES[i]: {
                "rmse":
                    float(
                        one_step[
                            "rmse"
                        ][i]
                    ),

                "mae":
                    float(
                        one_step[
                            "mae"
                        ][i]
                    ),

                "maxae":
                    float(
                        one_step[
                            "maxae"
                        ][i]
                    ),

                "fit_percent":
                    float(
                        one_step[
                            "fit"
                        ][i]
                    ),
            }
            for i in range(NX)
        },

        "free_run": {
            STATE_NAMES[i]: {
                "rmse":
                    float(
                        free_run[
                            "rmse"
                        ][i]
                    ),

                "mae":
                    float(
                        free_run[
                            "mae"
                        ][i]
                    ),

                "maxae":
                    float(
                        free_run[
                            "maxae"
                        ][i]
                    ),

                "fit_percent":
                    float(
                        free_run[
                            "fit"
                        ][i]
                    ),
            }
            for i in range(NX)
        },

        "results_file":
            result_npz,
    }

    summary_path = os.path.join(
        processed_dir,
        "pem_validation_summary.json"
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
    print(
        "Results:"
    )

    print(
        result_npz
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
        "[PASS] held-out dataset isolated"
    )

    print(
        "[PASS] one-step validation completed"
    )

    print(
        "[PASS] free-run validation completed"
    )


if __name__ == "__main__":
    main()
