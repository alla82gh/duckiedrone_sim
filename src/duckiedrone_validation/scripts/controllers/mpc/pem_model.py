#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_model.py

Frozen PEM prediction model for the Duckiedrone DD21.

The identified model is:

    x_c[k+1] = A_PEM x_c[k] + B_PEM u_c[k]

where the PEM coordinates are centered coordinates.

State convention
----------------

    x = [
        x, y, z,
        phi, theta, psi,
        vx, vy, vz,
        p, q, r
    ]

Nominal MPC input convention
----------------------------

    u_nominal = [
        T - T_hover,
        tau_phi,
        tau_theta,
        tau_psi
    ]

PEM input coordinates
---------------------

    u_c = u_nominal - u_trim_nominal

PEM state coordinates
---------------------

    x_c = x_absolute - x_op

The deployment operating point used here is the mean empirical
pre-hover trim of the FIVE TRAINING runs only.

No development-validation or final-test data are used to define
the deployment operating point.

The identified A and B matrices are loaded without any modification,
stability projection, clipping, or post-processing.

Author: Abdallah GHOUL 2026
"""

import hashlib
import os

import numpy as np

from .prediction_model import PredictionModel


class PEMModel(PredictionModel):
    """
    Frozen DD21 PEM prediction model.

    Internal model coordinates:

        x_c[k+1] = Ad @ x_c[k] + Bd @ u_c[k]

    where:

        x_c = x_absolute - x_op

        u_c = u_nominal - u_trim_nominal
    """

    # ============================================================
    # Frozen identified-model identity
    # ============================================================

    EXPECTED_SHA256 = (
        "3bbde4b14d80aca830a9b6bc197c5429"
        "b501a3d13702e49935e716505d2f67a1"
    )

    # Sampling time used during PEM identification.
    IDENTIFICATION_TS = 0.01

    # Nominal DD21 hover thrust used by pem_preprocess.py:
    #
    #     U_E = [6.22935, 0, 0, 0]
    #
    # This value belongs to the IDENTIFICATION coordinate
    # convention and must not silently change with runtime tuning.
    IDENTIFICATION_HOVER_THRUST = 6.22935

    # ============================================================
    # Training-only deployment operating point
    # ============================================================
    #
    # Mean empirical pre-hover state trim of:
    #
    #   pem_train_thrust_01
    #   pem_train_roll_01
    #   pem_train_pitch_01
    #   pem_train_yaw_01
    #   pem_train_mimo_01
    #
    # No validation or final-test run is included.
    # ============================================================

    TRAINING_X_OP = np.array(
        [
             5.20560954e-02,   # x
             9.79700946e-06,   # y
             1.46337534e+00,   # z

            -5.72813337e-07,   # phi
             8.73424760e-05,   # theta
             4.75363493e-09,   # psi

            -7.39900879e-04,   # vx
            -1.19315593e-05,   # vy
            -1.14998898e-03,   # vz

            -6.11294603e-07,   # p
            -1.89614702e-04,   # q
             1.22798434e-09,   # r
        ],
        dtype=float
    )

    # Mean pre-hover input trim in NOMINAL MPC coordinates:
    #
    #   u_nominal =
    #       [T - T_hover,
    #        tau_phi,
    #        tau_theta,
    #        tau_psi]
    #
    # and:
    #
    #   u_c = u_nominal - TRAINING_U_TRIM_NOMINAL
    #
    TRAINING_U_TRIM_NOMINAL = np.array(
        [
             4.00233417e-02,   # delta_T
             3.29539607e-08,   # tau_phi
            -7.35631352e-04,   # tau_theta
             2.52669156e-11,   # tau_psi
        ],
        dtype=float
    )

    EXPECTED_TRAINING_RUNS = (
        "pem_train_thrust_01",
        "pem_train_roll_01",
        "pem_train_pitch_01",
        "pem_train_yaw_01",
        "pem_train_mimo_01",
    )

    # ============================================================

    def __init__(
        self,
        parameters,
        model_path=None,
        verify_hash=True
    ):

        super().__init__(parameters)

        # Prediction horizon.
        self.N = int(self.params.Np)

        # --------------------------------------------------------
        # Resolve frozen model artifact
        # --------------------------------------------------------

        if model_path is None:

            this_dir = os.path.dirname(
                os.path.abspath(__file__)
            )

            # Current file:
            #
            #   duckiedrone_validation/
            #       scripts/
            #           controllers/
            #               mpc/
            #                   pem_model.py
            #
            # Package root is therefore three levels above mpc/.
            package_dir = os.path.abspath(
                os.path.join(
                    this_dir,
                    "..",
                    "..",
                    ".."
                )
            )

            model_path = os.path.join(
                package_dir,
                "data",
                "pem_identification",
                "processed",
                "pem_final_model.npz"
            )

        self.model_path = os.path.abspath(
            model_path
        )

        if not os.path.isfile(
            self.model_path
        ):
            raise RuntimeError(
                "Frozen PEM model artifact not found:\n"
                + self.model_path
            )

        # --------------------------------------------------------
        # Artifact integrity
        # --------------------------------------------------------

        self.artifact_sha256 = (
            self._sha256_file(
                self.model_path
            )
        )

        if verify_hash:

            if (
                self.artifact_sha256
                != self.EXPECTED_SHA256
            ):
                raise RuntimeError(
                    "PEM model SHA-256 mismatch.\n"
                    "Expected:\n"
                    f"  {self.EXPECTED_SHA256}\n"
                    "Received:\n"
                    f"  {self.artifact_sha256}\n"
                    "The frozen identified model must not "
                    "be modified."
                )

        # --------------------------------------------------------
        # Load frozen model
        # --------------------------------------------------------

        with np.load(
            self.model_path,
            allow_pickle=False
        ) as data:

            required = {
                "A",
                "B",
                "Ts",
            }

            missing = (
                required - set(data.files)
            )

            if missing:
                raise RuntimeError(
                    "PEM artifact is missing keys: "
                    + str(sorted(missing))
                )

            A = np.asarray(
                data["A"],
                dtype=float
            )

            B = np.asarray(
                data["B"],
                dtype=float
            )

            Ts = float(
                data["Ts"]
            )

            # Optional traceability metadata.
            self.model_type = (
                str(
                    np.asarray(
                        data["model_type"]
                    ).item()
                )
                if "model_type" in data.files
                else "unknown"
            )

            self.selection_status = (
                str(
                    np.asarray(
                        data["selection_status"]
                    ).item()
                )
                if "selection_status" in data.files
                else "unknown"
            )

            if "training_runs" in data.files:

                self.training_runs = tuple(
                    str(v)
                    for v in np.asarray(
                        data["training_runs"]
                    ).reshape(-1)
                )

            else:

                self.training_runs = tuple()

            self.development_validation = (
                str(
                    np.asarray(
                        data[
                            "development_validation"
                        ]
                    ).item()
                )
                if "development_validation"
                in data.files
                else "unknown"
            )

            self.saved_spectral_radius = (
                float(
                    data["spectral_radius"]
                )
                if "spectral_radius"
                in data.files
                else None
            )

        # --------------------------------------------------------
        # Validate frozen model dimensions
        # --------------------------------------------------------

        if A.shape != (
            self.nx,
            self.nx
        ):
            raise RuntimeError(
                "Invalid PEM A matrix shape: "
                f"{A.shape}; expected "
                f"({self.nx}, {self.nx})."
            )

        if B.shape != (
            self.nx,
            self.nu
        ):
            raise RuntimeError(
                "Invalid PEM B matrix shape: "
                f"{B.shape}; expected "
                f"({self.nx}, {self.nu})."
            )

        if not np.all(
            np.isfinite(A)
        ):
            raise RuntimeError(
                "PEM A matrix contains NaN or Inf."
            )

        if not np.all(
            np.isfinite(B)
        ):
            raise RuntimeError(
                "PEM B matrix contains NaN or Inf."
            )

        if Ts <= 0.0:
            raise RuntimeError(
                "Invalid PEM sampling time."
            )

        # The controller and identified model must use the
        # same sample period.
        if not np.isclose(
            Ts,
            float(self.params.Ts),
            atol=1.0e-12,
            rtol=0.0
        ):
            raise RuntimeError(
                "PEM sampling-time mismatch: "
                f"model Ts={Ts}, "
                f"controller Ts={self.params.Ts}."
            )

        if not np.isclose(
            Ts,
            self.IDENTIFICATION_TS,
            atol=1.0e-12,
            rtol=0.0
        ):
            raise RuntimeError(
                "Frozen PEM artifact does not use "
                "the expected identification Ts."
            )

        # --------------------------------------------------------
        # Validate training provenance if metadata exists
        # --------------------------------------------------------

        if self.training_runs:

            if set(self.training_runs) != set(
                self.EXPECTED_TRAINING_RUNS
            ):
                raise RuntimeError(
                    "PEM artifact training-run metadata "
                    "does not match the frozen training set."
                )

        # --------------------------------------------------------
        # Store discrete PEM model
        # --------------------------------------------------------
        #
        # Keep both naming conventions because the current
        # Physics MPC controller accesses:
        #
        #     self.model.Ad
        #     self.model.Bd
        #
        # directly for delay compensation and diagnostics.
        # --------------------------------------------------------

        self.A = A.copy()
        self.B = B.copy()

        self.Ad = self.A.copy()
        self.Bd = self.B.copy()

        self.Ts = Ts

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT project eigenvalues into the unit circle.
        # Do NOT clip or stabilize A.
        #
        # The model is intentionally the frozen final PEM model.
        # --------------------------------------------------------

        self.eigenvalues = np.linalg.eigvals(
            self.Ad
        )

        self.spectral_radius = float(
            np.max(
                np.abs(
                    self.eigenvalues
                )
            )
        )

        # --------------------------------------------------------
        # Fixed deployment coordinates
        # --------------------------------------------------------

        self.x_op = (
            self.TRAINING_X_OP.copy()
        )

        self.u_trim_nominal = (
            self.TRAINING_U_TRIM_NOMINAL.copy()
        )

        # Physical wrench corresponding to zero PEM input.
        #
        # u_model = 0
        #   ->
        # u_nominal = u_trim_nominal
        #   ->
        # u_physical =
        #       [T_hover, 0, 0, 0]
        #       + u_trim_nominal
        #
        self.u_base_physical = np.array(
            [
                self.IDENTIFICATION_HOVER_THRUST,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float
        )

        self.u_base_physical += (
            self.u_trim_nominal
        )

    # ============================================================
    # Artifact integrity
    # ============================================================

    @staticmethod
    def _sha256_file(path):
        """
        Compute SHA-256 of a file.
        """

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

    # ============================================================
    # PredictionModel interface
    # ============================================================

    def linearize(
        self,
        x,
        u
    ):
        """
        Return the frozen discrete PEM model.

        The PEM model is already linear and therefore does not
        depend on the current state or input.
        """

        return self.Ad, self.Bd

    # ------------------------------------------------------------

    def predict(
        self,
        x,
        u
    ):
        """
        One-step prediction in PEM-centered coordinates.

        Parameters
        ----------
        x : ndarray, shape (12,)
            Centered PEM state.

        u : ndarray, shape (4,)
            Centered PEM input.

        Returns
        -------
        ndarray, shape (12,)
            Predicted centered state.
        """

        x = np.asarray(
            x,
            dtype=float
        ).reshape(self.nx)

        u = np.asarray(
            u,
            dtype=float
        ).reshape(self.nu)

        return (
            self.Ad @ x
            + self.Bd @ u
        )

    # ------------------------------------------------------------

    def hover_input(self):
        """
        Return the equilibrium input in PEM MODEL coordinates.

        Zero PEM input corresponds to the empirical training
        hover trim.

        Returns
        -------
        ndarray, shape (4,)
        """

        return np.zeros(
            self.nu,
            dtype=float
        )

    # ============================================================
    # Prediction matrices
    # ============================================================

    def build_phi(self):
        """
        Build the state prediction matrix Phi.

        X = Phi x0 + Gamma U
        """

        Phi = np.zeros(
            (
                self.N * self.nx,
                self.nx
            ),
            dtype=float
        )

        for i in range(self.N):

            Phi[
                i * self.nx:
                (i + 1) * self.nx,
                :
            ] = np.linalg.matrix_power(
                self.Ad,
                i + 1
            )

        return Phi

    # ------------------------------------------------------------

    def build_gamma(self):
        """
        Build the control prediction matrix Gamma.
        """

        Gamma = np.zeros(
            (
                self.N * self.nx,
                self.N * self.nu
            ),
            dtype=float
        )

        for i in range(self.N):

            for j in range(i + 1):

                block = (
                    np.linalg.matrix_power(
                        self.Ad,
                        i - j
                    )
                    @ self.Bd
                )

                row = slice(
                    i * self.nx,
                    (i + 1) * self.nx
                )

                col = slice(
                    j * self.nu,
                    (j + 1) * self.nu
                )

                Gamma[
                    row,
                    col
                ] = block

        return Gamma

    # ------------------------------------------------------------

    def build_prediction_model(self):
        """
        Return prediction matrices.

        Returns
        -------
        Phi : ndarray
        Gamma : ndarray
        """

        return (
            self.build_phi(),
            self.build_gamma()
        )

    # ============================================================
    # State-coordinate transformations
    # ============================================================

    def to_model_state(
        self,
        x_absolute
    ):
        """
        Absolute controller state -> PEM-centered state.

            x_c = x_absolute - x_op
        """

        x_absolute = np.asarray(
            x_absolute,
            dtype=float
        ).reshape(self.nx)

        return (
            x_absolute
            - self.x_op
        )

    # ------------------------------------------------------------

    def to_absolute_state(
        self,
        x_model
    ):
        """
        PEM-centered state -> absolute controller state.

            x_absolute = x_c + x_op
        """

        x_model = np.asarray(
            x_model,
            dtype=float
        ).reshape(self.nx)

        return (
            x_model
            + self.x_op
        )

    # ------------------------------------------------------------

    def reference_to_model(
        self,
        x_ref
    ):
        """
        Convert an absolute reference to PEM coordinates.

        Supported:
            (nx,)
            (Np*nx,)
        """

        x_ref = np.asarray(
            x_ref,
            dtype=float
        ).reshape(-1)

        if x_ref.size == self.nx:

            return (
                x_ref - self.x_op
            )

        if x_ref.size == (
            self.N * self.nx
        ):

            X = x_ref.reshape(
                self.N,
                self.nx
            )

            Xc = (
                X
                - self.x_op[None, :]
            )

            return Xc.reshape(-1)

        raise ValueError(
            "PEM reference must have size "
            "nx or Np*nx."
        )

    # ============================================================
    # Input-coordinate transformations
    # ============================================================

    def nominal_to_model_input(
        self,
        u_nominal
    ):
        """
        Nominal MPC input -> centered PEM input.

        Nominal MPC:
            [T-T_hover, tau_phi, tau_theta, tau_psi]

        PEM:
            u_c = u_nominal - u_trim_nominal
        """

        u_nominal = np.asarray(
            u_nominal,
            dtype=float
        ).reshape(self.nu)

        return (
            u_nominal
            - self.u_trim_nominal
        )

    # ------------------------------------------------------------

    def model_to_nominal_input(
        self,
        u_model
    ):
        """
        Centered PEM input -> nominal MPC input.
        """

        u_model = np.asarray(
            u_model,
            dtype=float
        ).reshape(self.nu)

        return (
            u_model
            + self.u_trim_nominal
        )

    # ------------------------------------------------------------

    def physical_to_model_input(
        self,
        u_physical
    ):
        """
        Physical wrench -> centered PEM input.

        Physical:
            [T, tau_phi, tau_theta, tau_psi]
        """

        u_physical = np.asarray(
            u_physical,
            dtype=float
        ).reshape(self.nu)

        u_nominal = (
            u_physical.copy()
        )

        u_nominal[0] -= (
            self.IDENTIFICATION_HOVER_THRUST
        )

        return self.nominal_to_model_input(
            u_nominal
        )

    # ------------------------------------------------------------

    def model_to_physical_input(
        self,
        u_model
    ):
        """
        Centered PEM input -> physical wrench.
        """

        u_nominal = (
            self.model_to_nominal_input(
                u_model
            )
        )

        u_physical = (
            u_nominal.copy()
        )

        u_physical[0] += (
            self.IDENTIFICATION_HOVER_THRUST
        )

        return u_physical