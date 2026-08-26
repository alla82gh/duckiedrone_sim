#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
constraints.py

Constraint builder for the Physics-based MPC
controller of the Duckiedrone DD21.

The prediction model is:

    X = Phi @ x0 + Gamma @ U

The control increment model is:

    Delta_U = D @ U - b

The constraints are written in the standard
linear inequality form:

    A_ineq @ U <= b_ineq

The module supports:

1. Input constraints

       U_min <= U <= U_max

2. Control increment constraints

       Delta_U_min <= D @ U - b <= Delta_U_max

3. State constraints

       X_min <= Phi @ x0 + Gamma @ U <= X_max


State ordering:

    x = [
        x, y, z,
        phi, theta, psi,
        vx, vy, vz,
        p, q, r
    ]

Control ordering:

    u = [
        delta_T,
        tau_phi,
        tau_theta,
        tau_psi
    ]

Author: Abdallah GHOUL 2026
"""

import numpy as np

class Constraints:
    """
    Build linear inequality constraints
    for the Physics MPC quadratic program.
    """

    def __init__(self, parameters):

        self.params = parameters

        # --------------------------------------------------
        # Backward-compatible vehicle parameter view
        # --------------------------------------------------
        #
        # Historical validation code accesses:
        #
        #     constraints.vehicle.mass
        #     constraints.vehicle.gravity
        #
        # Use the SAME runtime MPCParameters instance rather
        # than creating a separate VehicleParameters object.
        # This preserves parameter-mismatch experiments.
        # --------------------------------------------------

        self.vehicle = parameters

        # --------------------------------------------------
        # Dimensions
        # --------------------------------------------------

        self.nx = parameters.nx
        self.nu = parameters.nu

        self.Np = parameters.Np
        self.Nc = parameters.Nc

        # --------------------------------------------------
        # Hover equilibrium
        # --------------------------------------------------

        self.hover_thrust = (
            float(parameters.mass)
            * float(parameters.gravity)
        )

        # --------------------------------------------------
        # Physical / control limits
        # --------------------------------------------------

        self.phi_max = parameters.phi_max
        self.theta_max = parameters.theta_max

        self.z_min = parameters.z_min
        self.z_max = parameters.z_max

        self.thrust_min = parameters.thrust_min
        self.thrust_max = parameters.thrust_max

        self.torque_max = parameters.torque_max

        self.du_max = parameters.du_max

        self.du_max_vec = np.asarray(
            parameters.du_max_vec,
            dtype=float
        )

        # --------------------------------------------------
        # Prediction-model coordinate offsets
        # --------------------------------------------------

        self.state_offset = np.asarray(
            parameters.state_offset,
            dtype=float
        ).reshape(self.nx)

        self.input_offset = np.asarray(
            parameters.input_offset,
            dtype=float
        ).reshape(self.nu)

        # --------------------------------------------------
        # Soft state constraints
        # --------------------------------------------------

        self.soft_state_constraints = bool(
            parameters.soft_state_constraints
        )

        self.slack_weight = float(
            parameters.slack_weight
        )

        # Soft-constrained state indices:
        #
        # 2 -> z
        # 3 -> phi
        # 4 -> theta
        self.soft_state_indices = (
            2,
            3,
            4
        )

        self.n_soft_states = len(
            self.soft_state_indices
        )

        self.n_slack = (
            self.Np * self.n_soft_states
            if self.soft_state_constraints
            else 0
        )

        # --------------------------------------------------
        # Rotor allocation parameters
        # --------------------------------------------------

        self.k_f = parameters.k_f
        self.k_m = parameters.k_m

        self.arm_dx = parameters.arm_dx
        self.arm_dy = parameters.arm_dy

        self.max_rotor_velocity = (
            parameters.max_rotor_velocity
        )

        # Forward allocation:
        #
        # [T, tau_phi, tau_theta, tau_psi]^T
        #       = M @ [w1^2, w2^2, w3^2, w4^2]^T
        #
        self.allocation_matrix = np.array(
            [
                [
                    self.k_f,
                    self.k_f,
                    self.k_f,
                    self.k_f
                ],
                [
                    -self.k_f * self.arm_dy,
                    -self.k_f * self.arm_dy,
                     self.k_f * self.arm_dy,
                     self.k_f * self.arm_dy
                ],
                [
                     self.k_f * self.arm_dx,
                    -self.k_f * self.arm_dx,
                     self.k_f * self.arm_dx,
                    -self.k_f * self.arm_dx
                ],
                [
                     self.k_m,
                    -self.k_m,
                    -self.k_m,
                     self.k_m
                ]
            ],
            dtype=float
        )

        self.allocation_inverse = np.linalg.inv(
            self.allocation_matrix
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if self.Np <= 0:
            raise ValueError(
                "Prediction horizon Np must be positive."
            )

        if self.Nc <= 0:
            raise ValueError(
                "Control horizon Nc must be positive."
            )

        if self.Nc != self.Np:
            raise ValueError(
                "Current prediction model requires Nc == Np."
            )

        if self.nx != 12:
            raise ValueError(
                "Current Physics MPC model requires nx = 12."
            )

        if self.nu != 4:
            raise ValueError(
                "Current Physics MPC model requires nu = 4."
            )

        if self.z_min >= self.z_max:
            raise ValueError(
                "z_min must be smaller than z_max."
            )

        if self.thrust_min >= self.thrust_max:
            raise ValueError(
                "thrust_min must be smaller than thrust_max."
            )

        if self.phi_max <= 0.0:
            raise ValueError(
                "phi_max must be positive."
            )

        if self.theta_max <= 0.0:
            raise ValueError(
                "theta_max must be positive."
            )

        if self.torque_max <= 0.0:
            raise ValueError(
                "torque_max must be positive."
            )

        if self.du_max <= 0.0:
            raise ValueError(
                "du_max must be positive."
            )

        if (
            self.soft_state_constraints
            and self.slack_weight <= 0.0
        ):
            raise ValueError(
                "slack_weight must be positive "
                "when soft state constraints are enabled."
            )

    # --------------------------------------------------

    def build_input_bounds(self):
        """
        Build stacked MPC input bounds.

        The MPC input vector is:

            u = [
                delta_T,
                tau_phi,
                tau_theta,
                tau_psi
            ]

        where:

            delta_T = T - T_hover

        and:

            T_hover = m * g

        Therefore the physical thrust limits:

            thrust_min <= T <= thrust_max

        become:

            thrust_min - T_hover
                <= delta_T
                <= thrust_max - T_hover

        Returns
        -------
        U_min : np.ndarray
            Lower stacked MPC input bounds.

        U_max : np.ndarray
            Upper stacked MPC input bounds.
        """

        # --------------------------------------------------
        # Bounds in nominal MPC coordinates
        #
        # u_nominal =
        #   [T-T_hover,
        #    tau_phi,
        #    tau_theta,
        #    tau_psi]
        # --------------------------------------------------

        u_nominal_min = np.array(
            [
                self.thrust_min
                - self.hover_thrust,

                -self.torque_max,
                -self.torque_max,
                -self.torque_max
            ],
            dtype=float
        )

        u_nominal_max = np.array(
            [
                self.thrust_max
                - self.hover_thrust,

                self.torque_max,
                self.torque_max,
                self.torque_max
            ],
            dtype=float
        )

        # --------------------------------------------------
        # Convert nominal MPC bounds to prediction-model
        # coordinates:
        #
        #     u_model =
        #         u_nominal - input_offset
        #
        # Physics MPC:
        #     input_offset = 0
        #
        # PEM-MPC:
        #     input_offset = u_trim_nominal
        # --------------------------------------------------

        u_min = (
            u_nominal_min
            - self.input_offset
        )

        u_max = (
            u_nominal_max
            - self.input_offset
        )

        # --------------------------------------------------
        # Stack over the control horizon
        # --------------------------------------------------

        U_min = np.tile(
            u_min,
            self.Nc
        )

        U_max = np.tile(
            u_max,
            self.Nc
        )

        return U_min, U_max

    # --------------------------------------------------

    def build_delta_bounds(self):
        """
        Build stacked control increment bounds.

        Per-channel symmetric increment bounds:

            -du_max_vec <= Delta_u <= du_max_vec

        Returns
        -------
        Delta_U_min : np.ndarray
            Lower control increment bounds.

        Delta_U_max : np.ndarray
            Upper control increment bounds.
        """

        delta_u_min = (
            -self.du_max_vec.copy()
        )

        delta_u_max = (
            self.du_max_vec.copy()
        )

        Delta_U_min = np.tile(
            delta_u_min,
            self.Nc
        )

        Delta_U_max = np.tile(
            delta_u_max,
            self.Nc
        )

        return Delta_U_min, Delta_U_max

    # --------------------------------------------------

    def build_state_bounds(self):
        """
        Build stacked state bounds.

        State ordering:

            [
                x, y, z,
                phi, theta, psi,
                vx, vy, vz,
                p, q, r
            ]

        Currently constrained:

            z
            phi
            theta

        Other states remain unbounded.

        Returns
        -------
        X_min : np.ndarray
            Lower stacked state bounds.

        X_max : np.ndarray
            Upper stacked state bounds.
        """

        x_min_absolute = np.array(
            [
                -np.inf,              # x
                -np.inf,              # y
                self.z_min,           # z

                -self.phi_max,        # phi
                -self.theta_max,      # theta
                -np.inf,              # psi

                -np.inf,              # vx
                -np.inf,              # vy
                -np.inf,              # vz

                -np.inf,              # p
                -np.inf,              # q
                -np.inf               # r
            ],
            dtype=float
        )

        x_max_absolute = np.array(
            [
                np.inf,               # x
                np.inf,               # y
                self.z_max,           # z

                self.phi_max,         # phi
                self.theta_max,       # theta
                np.inf,               # psi

                np.inf,               # vx
                np.inf,               # vy
                np.inf,               # vz

                np.inf,               # p
                np.inf,               # q
                np.inf                # r
            ],
            dtype=float
        )

        # --------------------------------------------------
        # Convert physical absolute-state limits to
        # prediction-model coordinates:
        #
        #     x_model =
        #         x_absolute - state_offset
        #
        # Infinite entries remain infinite.
        # --------------------------------------------------

        x_min = (
            x_min_absolute
            - self.state_offset
        )

        x_max = (
            x_max_absolute
            - self.state_offset
        )

        X_min = np.tile(
            x_min,
            self.Np
        )

        X_max = np.tile(
            x_max,
            self.Np
        )

        return X_min, X_max

    # --------------------------------------------------

    def build_input_inequalities(self):
        """
        Convert input bounds into:

            A_u @ U <= b_u

        From:

            U <= U_max

           -U <= -U_min

        Returns
        -------
        A_u : np.ndarray
            Input inequality matrix.

        b_u : np.ndarray
            Input inequality vector.
        """

        U_min, U_max = (
            self.build_input_bounds()
        )

        size = self.Nc * self.nu

        I = np.eye(
            size,
            dtype=float
        )

        A_u = np.vstack(
            (
                I,
                -I
            )
        )

        b_u = np.concatenate(
            (
                U_max,
                -U_min
            )
        )

        return A_u, b_u

    # --------------------------------------------------

    def build_rotor_inequalities(self):
        """
        Build rotor-feasibility constraints.

        The physical wrench is:

            u_physical =
                u_hover + u_mpc

        where:

            u_hover =
                [T_hover, 0, 0, 0]

            u_mpc =
                [delta_T,
                 tau_phi,
                 tau_theta,
                 tau_psi]

        Rotor squared velocities satisfy:

            w2 = M^{-1} @ u_physical

        Physical actuator feasibility requires:

            0 <= w2 <= w_max^2

        Therefore:

            M^{-1} u_mpc
                <= w_max^2 - M^{-1} u_hover

           -M^{-1} u_mpc
                <= M^{-1} u_hover

        Returns
        -------
        A_rotor : np.ndarray
            Rotor feasibility inequality matrix.

        b_rotor : np.ndarray
            Rotor feasibility upper-bound vector.
        """

        # --------------------------------------------------
        # Physical wrench corresponding to zero MODEL input
        #
        # Coordinate relation:
        #
        #     u_nominal =
        #         u_model + input_offset
        #
        # Physical wrench:
        #
        #     u_physical =
        #         [T_hover, 0, 0, 0]
        #         + input_offset
        #         + u_model
        #
        # Therefore zero model input corresponds to:
        #
        #     u_base_physical =
        #         [T_hover, 0, 0, 0]
        #         + input_offset
        #
        # Physics MPC:
        #     input_offset = 0
        #
        # PEM-MPC:
        #     input_offset = u_trim_nominal
        # --------------------------------------------------

        u_base_physical = np.array(
            [
                self.hover_thrust,
                0.0,
                0.0,
                0.0
            ],
            dtype=float
        )

        u_base_physical += (
            self.input_offset
        )

        # Squared rotor velocities at zero model input
        w2_base = (
            self.allocation_inverse
            @ u_base_physical
        )

        # Maximum squared rotor velocity
        w2_max = (
            self.max_rotor_velocity ** 2
        )

        # Stacked mapping:
        #
        # [u0, u1, ..., uNc-1]
        #       ->
        # [w2_0, w2_1, ..., w2_Nc-1]
        K = np.kron(
            np.eye(self.Nc),
            self.allocation_inverse
        )

        upper_single = (
            w2_max * np.ones(
                self.nu,
                dtype=float
            )
            - w2_base
        )

        lower_single = (
            w2_base.copy()
        )

        A_rotor = np.vstack(
            (
                K,
                -K
            )
        )

        b_rotor = np.concatenate(
            (
                np.tile(
                    upper_single,
                    self.Nc
                ),
                np.tile(
                    lower_single,
                    self.Nc
                )
            )
        )

        return A_rotor, b_rotor

    # --------------------------------------------------

    def build_delta_inequalities(
        self,
        D,
        u_prev
    ):
        """
        Build control increment inequalities.

        Starting from:

            Delta_U = D @ U - b

        with:

            Delta_U_min
                <= D @ U - b
                <= Delta_U_max

        the inequalities become:

            D @ U <= Delta_U_max + b

           -D @ U <= -Delta_U_min - b

        Parameters
        ----------
        D : np.ndarray
            Control difference matrix.

        u_prev : np.ndarray
            Previously applied control input.

        Returns
        -------
        A_du : np.ndarray
            Control increment inequality matrix.

        b_du : np.ndarray
            Control increment inequality vector.
        """

        size = self.Nc * self.nu

        D = np.asarray(
            D,
            dtype=float
        )

        if D.shape != (size, size):
            raise ValueError(
                f"D must have shape "
                f"({size}, {size})."
            )

        u_prev = np.asarray(
            u_prev,
            dtype=float
        ).reshape(self.nu)

        # --------------------------------------------------
        # Delta bounds
        # --------------------------------------------------

        Delta_U_min, Delta_U_max = (
            self.build_delta_bounds()
        )

        # --------------------------------------------------
        # Offset vector
        #
        # b = [u_prev, 0, ..., 0]
        # --------------------------------------------------

        b = np.zeros(
            size,
            dtype=float
        )

        b[:self.nu] = u_prev

        # --------------------------------------------------
        # Inequality matrices
        # --------------------------------------------------

        A_du = np.vstack(
            (
                D,
                -D
            )
        )

        b_du = np.concatenate(
            (
                Delta_U_max + b,
                -Delta_U_min - b
            )
        )

        return A_du, b_du

    # --------------------------------------------------

    def build_state_inequalities(
        self,
        Phi,
        Gamma,
        x0
    ):
        """
        Build state inequalities.

        Prediction:

            X = Phi @ x0 + Gamma @ U

        Constraints:

            X_min <= X <= X_max

        Upper bound:

            Gamma @ U
                <= X_max - Phi @ x0

        Lower bound:

           -Gamma @ U
                <= Phi @ x0 - X_min

        Infinite state bounds are removed
        from the final inequality system.

        Parameters
        ----------
        Phi : np.ndarray
            State prediction matrix.

        Gamma : np.ndarray
            Control prediction matrix.

        x0 : np.ndarray
            Current system state.

        Returns
        -------
        A_x : np.ndarray
            State inequality matrix.

        b_x : np.ndarray
            State inequality vector.
        """

        expected_phi_shape = (
            self.Np * self.nx,
            self.nx
        )

        expected_gamma_shape = (
            self.Np * self.nx,
            self.Nc * self.nu
        )

        Phi = np.asarray(
            Phi,
            dtype=float
        )

        Gamma = np.asarray(
            Gamma,
            dtype=float
        )

        if Phi.shape != expected_phi_shape:
            raise ValueError(
                f"Phi must have shape "
                f"{expected_phi_shape}."
            )

        if Gamma.shape != expected_gamma_shape:
            raise ValueError(
                f"Gamma must have shape "
                f"{expected_gamma_shape}."
            )

        x0 = np.asarray(
            x0,
            dtype=float
        ).reshape(self.nx)

        # --------------------------------------------------
        # State bounds
        # --------------------------------------------------

        X_min, X_max = (
            self.build_state_bounds()
        )

        # --------------------------------------------------
        # Free response
        # --------------------------------------------------

        X_free = Phi @ x0

        # --------------------------------------------------
        # Finite upper bounds
        # --------------------------------------------------

        upper_mask = np.isfinite(
            X_max
        )

        A_upper = Gamma[
            upper_mask,
            :
        ]

        b_upper = (
            X_max[upper_mask]
            - X_free[upper_mask]
        )

        # --------------------------------------------------
        # Finite lower bounds
        # --------------------------------------------------

        lower_mask = np.isfinite(
            X_min
        )

        A_lower = -Gamma[
            lower_mask,
            :
        ]

        b_lower = (
            X_free[lower_mask]
            - X_min[lower_mask]
        )

        # --------------------------------------------------
        # Stack
        # --------------------------------------------------

        A_x = np.vstack(
            (
                A_upper,
                A_lower
            )
        )

        b_x = np.concatenate(
            (
                b_upper,
                b_lower
            )
        )

        return A_x, b_x

    # --------------------------------------------------

    def get_num_slack_variables(self):
        """
        Return the number of soft-state slack variables.

        Slack ordering:

            [eps_z_1, eps_phi_1, eps_theta_1,
             eps_z_2, eps_phi_2, eps_theta_2,
             ...
             eps_z_Np, eps_phi_Np, eps_theta_Np]

        Returns
        -------
        int
            Number of slack variables.
        """

        return self.n_slack

    # --------------------------------------------------

    def build_soft_state_inequalities(
        self,
        Phi,
        Gamma,
        x0
    ):
        """
        Build softened state inequalities.

        Original hard state constraints:

            A_x @ U <= b_x

        become:

            A_x @ U - E @ epsilon <= b_x

        where:

            epsilon >= 0

        Decision vector:

            Z = [U ; epsilon]

        Returns
        -------
        A_soft : np.ndarray
            Soft-state inequality matrix.

        b_soft : np.ndarray
            Soft-state upper-bound vector.
        """

        if not self.soft_state_constraints:
            raise RuntimeError(
                "Soft state constraints are disabled."
            )

        A_x, b_x = (
            self.build_state_inequalities(
                Phi,
                Gamma,
                x0
            )
        )

        nU = (
            self.Nc
            * self.nu
        )

        n_eps = self.n_slack

        # State inequality structure:
        #
        # upper:
        #   z, phi, theta for every prediction step
        #
        # lower:
        #   z, phi, theta for every prediction step
        #
        # Therefore:
        #
        # rows = 2 * 3 * Np
        #      = 2 * n_eps

        expected_shape = (
            2 * n_eps,
            nU
        )

        if A_x.shape != expected_shape:
            raise ValueError(
                "Unexpected state-constraint shape "
                f"{A_x.shape}; expected "
                f"{expected_shape}."
            )

        # One slack variable per constrained state
        # and prediction step.
        #
        # The same epsilon relaxes both:
        #
        #   upper bound
        #   lower bound

        E = np.eye(
            n_eps,
            dtype=float
        )

        E_soft = np.vstack(
            (
                E,
                E
            )
        )

        A_soft = np.hstack(
            (
                A_x,
                -E_soft
            )
        )

        return A_soft, b_x

    # --------------------------------------------------

    def build_slack_inequalities(self):
        """
        Build slack non-negativity constraints.

        epsilon >= 0

        written as:

            -epsilon <= 0

        Decision vector:

            Z = [U ; epsilon]

        Returns
        -------
        A_eps : np.ndarray
            Slack inequality matrix.

        b_eps : np.ndarray
            Slack inequality upper bound.
        """

        if not self.soft_state_constraints:
            raise RuntimeError(
                "Soft state constraints are disabled."
            )

        nU = (
            self.Nc
            * self.nu
        )

        n_eps = self.n_slack

        A_eps = np.hstack(
            (
                np.zeros(
                    (n_eps, nU),
                    dtype=float
                ),
                -np.eye(
                    n_eps,
                    dtype=float
                )
            )
        )

        b_eps = np.zeros(
            n_eps,
            dtype=float
        )

        return A_eps, b_eps

    # --------------------------------------------------

    def build_qp_inequalities(
        self,
        Phi,
        Gamma,
        x0,
        u_prev,
        D
    ):
        """
        Build the complete QP inequality system:

            A_ineq @ U <= b_ineq

        including:

            - input constraints
            - control increment constraints
            - state constraints

        Parameters
        ----------
        Phi : np.ndarray
            State prediction matrix.

        Gamma : np.ndarray
            Control prediction matrix.

        x0 : np.ndarray
            Current state.

        u_prev : np.ndarray
            Previously applied control input.

        D : np.ndarray
            Control difference matrix.

        Returns
        -------
        A_ineq : np.ndarray
            Complete inequality matrix.

        b_ineq : np.ndarray
            Complete inequality vector.
        """

        # --------------------------------------------------
        # Input constraints
        # --------------------------------------------------

        A_u, b_u = (
            self.build_input_inequalities()
        )

        # --------------------------------------------------
        # Delta-U constraints
        # --------------------------------------------------

        A_du, b_du = (
            self.build_delta_inequalities(
                D,
                u_prev
            )
        )

        # --------------------------------------------------
        # Rotor feasibility constraints
        # --------------------------------------------------

        A_rotor, b_rotor = (
            self.build_rotor_inequalities()
        )

        # --------------------------------------------------
        # State constraints
        # --------------------------------------------------

        A_x, b_x = (
            self.build_state_inequalities(
                Phi,
                Gamma,
                x0
            )
        )

        # --------------------------------------------------
        # Complete inequality system
        # --------------------------------------------------

        A_ineq = np.vstack(
            (
                A_u,
                A_du,
                A_rotor,
                A_x
            )
        )

        b_ineq = np.concatenate(
            (
                b_u,
                b_du,
                b_rotor,
                b_x
            )
        )

        return A_ineq, b_ineq

    # --------------------------------------------------

    def build_soft_qp_inequalities(
        self,
        Phi,
        Gamma,
        x0,
        u_prev,
        D
    ):
        """
        Build the augmented QP inequality system
        for soft state constraints.

        Decision vector:

            Z = [U ; epsilon]

        Hard constraints:

            input
            Delta-U
            rotor feasibility

        Soft constraints:

            z
            phi
            theta

        Slack constraint:

            epsilon >= 0

        Returns
        -------
        A_ineq : np.ndarray
            Augmented inequality matrix.

        b_ineq : np.ndarray
            Augmented inequality upper bound.
        """

        if not self.soft_state_constraints:
            raise RuntimeError(
                "Soft state constraints are disabled."
            )

        n_eps = self.n_slack

        # --------------------------------------------------
        # Hard input constraints
        # --------------------------------------------------

        A_u, b_u = (
            self.build_input_inequalities()
        )

        # --------------------------------------------------
        # Hard Delta-U constraints
        # --------------------------------------------------

        A_du, b_du = (
            self.build_delta_inequalities(
                D,
                u_prev
            )
        )

        # --------------------------------------------------
        # Hard rotor-feasibility constraints
        # --------------------------------------------------

        A_rotor, b_rotor = (
            self.build_rotor_inequalities()
        )

        # --------------------------------------------------
        # Soft state constraints
        # --------------------------------------------------

        A_soft, b_soft = (
            self.build_soft_state_inequalities(
                Phi,
                Gamma,
                x0
            )
        )

        # --------------------------------------------------
        # Slack non-negativity
        # --------------------------------------------------

        A_eps, b_eps = (
            self.build_slack_inequalities()
        )

        # --------------------------------------------------
        # Add zero slack columns to hard constraints.
        #
        # Hard constraints depend only on U.
        # --------------------------------------------------

        def augment_hard(A):

            return np.hstack(
                (
                    A,
                    np.zeros(
                        (
                            A.shape[0],
                            n_eps
                        ),
                        dtype=float
                    )
                )
            )

        A_u_aug = augment_hard(
            A_u
        )

        A_du_aug = augment_hard(
            A_du
        )

        A_rotor_aug = augment_hard(
            A_rotor
        )

        # --------------------------------------------------
        # Complete augmented inequality system
        # --------------------------------------------------

        A_ineq = np.vstack(
            (
                A_u_aug,
                A_du_aug,
                A_rotor_aug,
                A_soft,
                A_eps
            )
        )

        b_ineq = np.concatenate(
            (
                b_u,
                b_du,
                b_rotor,
                b_soft,
                b_eps
            )
        )

        return A_ineq, b_ineq