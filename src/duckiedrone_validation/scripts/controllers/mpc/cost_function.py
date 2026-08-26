#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cost_function.py

Quadratic cost function for the Physics-based MPC
controller of the Duckiedrone DD21.

The cost function is written in the standard QP form:

    J(U) = 0.5 * U.T @ H @ U + f.T @ U

with the prediction model:

    X = Phi @ x0 + Gamma @ U

Author: Abdallah GHOUL 2026
"""

import numpy as np


class CostFunction:
    """
    Build the quadratic MPC cost matrices.
    """

    def __init__(self, parameters):

        self.params = parameters

        # System dimensions
        self.nx = parameters.nx
        self.nu = parameters.nu

        # Prediction and control horizons
        self.Np = parameters.Np
        self.Nc = parameters.Nc

        # Cost weights
        self.Q = parameters.Q
        self.R = parameters.R
        self.S = parameters.S
        self.P = parameters.P

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if self.Np <= 0:
            raise ValueError("Prediction horizon Np must be positive.")

        if self.Nc <= 0:
            raise ValueError("Control horizon Nc must be positive.")

        if self.Nc != self.Np:
            raise ValueError(
                "Current prediction model requires Nc == Np."
            )

        if self.Q.shape != (self.nx, self.nx):
            raise ValueError("Q has invalid dimensions.")

        if self.R.shape != (self.nu, self.nu):
            raise ValueError("R has invalid dimensions.")

        if self.S.shape != (self.nu, self.nu):
            raise ValueError("S has invalid dimensions.")

        if self.P.shape != (self.nx, self.nx):
            raise ValueError("P has invalid dimensions.")

    # --------------------------------------------------

    def build_qbar(self):
        """
        Build the augmented state weighting matrix Qbar.

        Qbar = diag(Q, Q, ..., Q, P)

        Returns
        -------
        np.ndarray
            Augmented state weighting matrix.
        """

        Qbar = np.zeros(
            (self.Np * self.nx,
             self.Np * self.nx),
            dtype=float
        )

        # Stage state weights
        for i in range(self.Np - 1):

            row = slice(
                i * self.nx,
                (i + 1) * self.nx
            )

            Qbar[row, row] = self.Q

        # Terminal state weight
        terminal = slice(
            (self.Np - 1) * self.nx,
            self.Np * self.nx
        )

        Qbar[terminal, terminal] = self.P

        return Qbar

    # --------------------------------------------------

    def build_rbar(self):
        """
        Build the augmented control weighting matrix Rbar.

        Rbar = diag(R, R, ..., R)

        Returns
        -------
        np.ndarray
            Augmented control weighting matrix.
        """

        Rbar = np.kron(
            np.eye(self.Nc),
            self.R
        )

        return Rbar

    # --------------------------------------------------

    def build_sbar(self):
        """
        Build the augmented control increment weighting matrix Sbar.

        Sbar = diag(S, S, ..., S)

        Returns
        -------
        np.ndarray
            Augmented control increment weighting matrix.
        """

        Sbar = np.kron(
            np.eye(self.Nc),
            self.S
        )

        return Sbar

    # --------------------------------------------------

    def build_delta_matrix(self):
        """
        Build the control increment matrix D.

        The stacked control increment vector is:

            Delta_U = D @ U - b

        where:

            Delta_u[0] = u[0] - u_prev

            Delta_u[k] = u[k] - u[k-1],
            k = 1, ..., Nc-1

        Returns
        -------
        np.ndarray
            Difference matrix D.
        """

        size = self.Nc * self.nu

        D = np.zeros(
            (size, size),
            dtype=float
        )

        Iu = np.eye(
            self.nu,
            dtype=float
        )

        for k in range(self.Nc):

            row = slice(
                k * self.nu,
                (k + 1) * self.nu
            )

            # Current input u[k]
            D[row, row] = Iu

            # Previous predicted input u[k-1]
            if k > 0:

                previous = slice(
                    (k - 1) * self.nu,
                    k * self.nu
                )

                D[row, previous] = -Iu

        return D

    # --------------------------------------------------

    def build_delta_offset(self, u_prev):
        """
        Build the control increment offset vector b.

        Delta_U = D @ U - b

        with:

            b = [u_prev, 0, ..., 0]^T

        Parameters
        ----------
        u_prev : np.ndarray
            Previously applied control input.

        Returns
        -------
        np.ndarray
            Offset vector b.
        """

        u_prev = np.asarray(
            u_prev,
            dtype=float
        ).reshape(self.nu)

        b = np.zeros(
            self.Nc * self.nu,
            dtype=float
        )

        b[:self.nu] = u_prev

        return b

    # --------------------------------------------------

    def build_reference(self, x_ref):
        """
        Build the stacked state reference vector.

        Parameters
        ----------
        x_ref : np.ndarray
            Either:
            - one state reference of size nx
            - full reference trajectory of size Np*nx

        Returns
        -------
        np.ndarray
            Stacked reference vector X_ref.
        """

        x_ref = np.asarray(
            x_ref,
            dtype=float
        ).reshape(-1)

        # Constant reference over the horizon
        if x_ref.size == self.nx:

            X_ref = np.tile(
                x_ref,
                self.Np
            )

            return X_ref

        # Already stacked reference trajectory
        if x_ref.size == self.Np * self.nx:

            return x_ref

        raise ValueError(
            "x_ref must have size nx or Np*nx."
        )

    # --------------------------------------------------

    def build_qp_matrices(
        self,
        Phi,
        Gamma,
        x0,
        x_ref,
        u_prev
    ):
        """
        Build the QP Hessian H and gradient f.

        Cost:

            J(U) = 0.5 * U.T @ H @ U + f.T @ U

        Parameters
        ----------
        Phi : np.ndarray
            State prediction matrix.

        Gamma : np.ndarray
            Control prediction matrix.

        x0 : np.ndarray
            Current state.

        x_ref : np.ndarray
            State reference or stacked reference trajectory.

        Returns
        -------
        H : np.ndarray
            Hessian matrix.

        f : np.ndarray
            Gradient vector.
        """

        x0 = np.asarray(
            x0,
            dtype=float
        ).reshape(self.nx)

        X_ref = self.build_reference(x_ref)
        u_prev = np.asarray(
        u_prev,
        dtype=float
        ).reshape(self.nu)

        Qbar = self.build_qbar()
        Rbar = self.build_rbar()
        Sbar = self.build_sbar()

        D = self.build_delta_matrix()
        b = self.build_delta_offset(u_prev)

        # --------------------------------------------------
        # Free prediction error
        # --------------------------------------------------

        e0 = Phi @ x0 - X_ref

        # --------------------------------------------------
        # QP Hessian
        # --------------------------------------------------

        H = 2.0 * (
            Gamma.T @ Qbar @ Gamma
            + Rbar
            + D.T @ Sbar @ D
        )

        # Numerical symmetry
        H = 0.5 * (H + H.T)

        # --------------------------------------------------
        # QP gradient
        # --------------------------------------------------

        f = (
            2.0 * (
                Gamma.T @ Qbar @ e0
            )
            -
            2.0 * (
                D.T @ Sbar @ b
            )
        )

        return H, f

    # --------------------------------------------------

    def build_soft_qp_matrices(
        self,
        Phi,
        Gamma,
        x0,
        x_ref,
        u_prev,
        n_slack
    ):
        """
        Build the augmented QP matrices for soft state constraints.

        The augmented decision vector is:

            Z = [U; epsilon]

        where:

            U       : stacked MPC control inputs
            epsilon : non-negative slack variables

        The augmented cost is:

            J(Z) =
                0.5 * Z.T @ H_soft @ Z
                + f_soft.T @ Z

        with:

            H_soft =
            [ H_U                  0 ]
            [  0      2*rho_eps*I  ]

        and:

            f_soft = [f_U; 0]

        The original MPC cost matrices H_U and f_U are generated
        by build_qp_matrices() without modification.

        Parameters
        ----------
        Phi : np.ndarray
            State prediction matrix.

        Gamma : np.ndarray
            Control prediction matrix.

        x0 : np.ndarray
            Current state.

        x_ref : np.ndarray
            State reference or stacked reference trajectory.

        u_prev : np.ndarray
            Previously applied control input.

        n_slack : int
            Number of slack variables.

        Returns
        -------
        H_soft : np.ndarray
            Augmented Hessian matrix.

        f_soft : np.ndarray
            Augmented gradient vector.
        """

        # --------------------------------------------------
        # Validate slack dimension
        # --------------------------------------------------

        if not isinstance(n_slack, (int, np.integer)):
            raise TypeError(
                "n_slack must be an integer."
            )

        if n_slack <= 0:
            raise ValueError(
                "n_slack must be positive."
            )

        # --------------------------------------------------
        # Original hard-QP cost
        # --------------------------------------------------

        H_U, f_U = self.build_qp_matrices(
            Phi=Phi,
            Gamma=Gamma,
            x0=x0,
            x_ref=x_ref,
            u_prev=u_prev
        )

        n_u = self.Nc * self.nu

        # --------------------------------------------------
        # Slack penalty
        # --------------------------------------------------

        rho_eps = float(
            self.params.slack_weight
        )

        if rho_eps <= 0.0:
            raise ValueError(
                "slack_weight must be positive."
            )

        # --------------------------------------------------
        # Augmented Hessian
        # --------------------------------------------------

        n_z = n_u + n_slack

        H_soft = np.zeros(
            (n_z, n_z),
            dtype=float
        )

        # Original control-input cost
        H_soft[
            :n_u,
            :n_u
        ] = H_U

        # Slack-variable quadratic penalty
        H_soft[
            n_u:,
            n_u:
        ] = (
            2.0
            * rho_eps
            * np.eye(
                n_slack,
                dtype=float
            )
        )

        # --------------------------------------------------
        # Augmented gradient
        # --------------------------------------------------

        f_soft = np.zeros(
            n_z,
            dtype=float
        )

        f_soft[:n_u] = f_U

        return H_soft, f_soft