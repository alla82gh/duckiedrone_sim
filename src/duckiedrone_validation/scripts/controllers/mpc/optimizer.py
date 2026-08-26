#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimizer.py

OSQP-based quadratic programming optimizer
for the Physics-based MPC controller
of the Duckiedrone DD21.

The MPC optimization problem is:

    minimize

        J(U) = 0.5 * U.T @ H @ U
               + f.T @ U

    subject to

        A_ineq @ U <= b_ineq

OSQP uses the standard form:

    minimize

        0.5 * U.T @ P @ U
        + q.T @ U

    subject to

        l <= A @ U <= u

Therefore:

    P = H
    q = f

    A = A_ineq
    l = -inf
    u = b_ineq

The Hessian and constraint matrix are constant
for the current linear Physics MPC formulation.

Only the following terms are updated online:

    q = f(x0, x_ref, u_prev)

    u = b_ineq(x0, u_prev)

This allows the OSQP workspace to be created
once and reused at every MPC control step.

Author: Abdallah GHOUL 2026
"""

import numpy as np
import osqp

from scipy import sparse

from .cost_function import CostFunction
from .constraints import Constraints


class Optimizer:
    """
    OSQP solver for the Physics MPC quadratic program.
    """

    def __init__(
        self,
        parameters,
        Phi,
        Gamma
    ):

        self.params = parameters

        # --------------------------------------------------
        # Dimensions
        # --------------------------------------------------

        self.nx = parameters.nx
        self.nu = parameters.nu

        self.Np = parameters.Np
        self.Nc = parameters.Nc

        self.nU = (
            self.Nc
            * self.nu
        )

        # --------------------------------------------------
        # Soft state constraints
        # --------------------------------------------------

        self.soft_state_constraints = bool(
            parameters.soft_state_constraints
        )

        # Initialized after Constraints is created
        self.n_slack = 0

        # Total optimization decision dimension
        self.nZ = self.nU

        # --------------------------------------------------
        # Solver settings
        # --------------------------------------------------

        self.max_iterations = (
            parameters.max_iterations
        )

        self.tolerance = (
            parameters.tolerance
        )

        # --------------------------------------------------
        # Prediction matrices
        # --------------------------------------------------

        self.Phi = np.asarray(
            Phi,
            dtype=float
        )

        self.Gamma = np.asarray(
            Gamma,
            dtype=float
        )

        # --------------------------------------------------
        # Validate prediction matrices
        # --------------------------------------------------

        expected_phi_shape = (
            self.Np * self.nx,
            self.nx
        )

        expected_gamma_shape = (
            self.Np * self.nx,
            self.Nc * self.nu
        )

        if self.Phi.shape != expected_phi_shape:

            raise ValueError(
                f"Phi must have shape "
                f"{expected_phi_shape}, "
                f"got {self.Phi.shape}."
            )

        if self.Gamma.shape != expected_gamma_shape:

            raise ValueError(
                f"Gamma must have shape "
                f"{expected_gamma_shape}, "
                f"got {self.Gamma.shape}."
            )

        # --------------------------------------------------
        # MPC components
        # --------------------------------------------------

        self.cost = CostFunction(
            parameters
        )

        self.constraints = Constraints(
            parameters
        )

        # --------------------------------------------------
        # Augmented decision-vector dimensions
        #
        # Hard QP:
        #     U in R^(Nc*nu)
        #
        # Soft QP:
        #     Z = [U; epsilon]
        # --------------------------------------------------

        if self.soft_state_constraints:

            self.n_slack = (
                self.constraints.get_num_slack_variables()
            )

            self.nZ = (
                self.nU
                + self.n_slack
            )

        # --------------------------------------------------
        # Delta-U matrix
        # --------------------------------------------------

        self.D = (
            self.cost.build_delta_matrix()
        )

        # --------------------------------------------------
        # Cached weighting matrices
        # --------------------------------------------------

        self.Qbar = (
            self.cost.build_qbar()
        )

        self.Sbar = (
            self.cost.build_sbar()
        )

        # --------------------------------------------------
        # OSQP solver
        # --------------------------------------------------

        self.solver = None

        self.initialized = False

        # --------------------------------------------------
        # Cached QP data
        # --------------------------------------------------

        self.H = None

        self.A_ineq = None

        self.lower_bound = None

        # Positive row multipliers used only for
        # numerical conditioning of OSQP constraints.
        #
        # The physical feasible set is unchanged.
        self.constraint_row_scaling = None

        # --------------------------------------------------
        # Last solution information
        # --------------------------------------------------

        self.last_solution = np.zeros(
            self.nU,
            dtype=float
        )

        self.last_slack_solution = np.zeros(
            self.n_slack,
            dtype=float
        )

        self.last_status = None

        self.last_objective = None

        self.last_iterations = None

        self.last_solve_time = None

        self.last_max_iter_hit = False

        # --------------------------------------------------
        # Build solver workspace
        # --------------------------------------------------

        self._initialize_solver()

    # --------------------------------------------------

    def _build_constraint_row_scaling(
        self,
        A_ineq
    ):
        """
        Build positive row multipliers for numerical
        conditioning of the OSQP constraint matrix.

        Only the rotor-feasibility rows are scaled.

        Original inequality:

            A_i @ Z <= b_i

        is replaced by:

            s_i * A_i @ Z <= s_i * b_i

        with:

            s_i > 0

        Therefore the feasible set is unchanged.

        Rotor rows are normalized by their infinity norm
        when that norm is greater than one.
        """

        A_ineq = np.asarray(
            A_ineq,
            dtype=float
        )

        n_rows = A_ineq.shape[0]

        scaling = np.ones(
            n_rows,
            dtype=float
        )

        # --------------------------------------------------
        # Row layout
        #
        # Input:
        #     0 ... 2*nU-1
        #
        # Delta-U:
        #     2*nU ... 4*nU-1
        #
        # Rotor:
        #     4*nU ... 6*nU-1
        # --------------------------------------------------

        rotor_start = (
            4
            * self.nU
        )

        rotor_end = (
            6
            * self.nU
        )

        if rotor_end > n_rows:

            raise ValueError(
                "Invalid constraint layout for "
                "rotor row scaling."
            )

        A_rotor = A_ineq[
            rotor_start:
            rotor_end,
            :
        ]

        rotor_row_norms = np.max(
            np.abs(A_rotor),
            axis=1
        )

        if np.any(
            rotor_row_norms <= 0.0
        ):

            raise ValueError(
                "Rotor constraint contains "
                "a zero row."
            )

        # Do not amplify rows whose norm is already <= 1.
        rotor_scaling = (
            1.0
            / np.maximum(
                rotor_row_norms,
                1.0
            )
        )

        scaling[
            rotor_start:
            rotor_end
        ] = rotor_scaling

        return scaling

    # --------------------------------------------------

    def _initialize_solver(self):
        """
        Build the constant part of the OSQP problem.

        This method is called only once.

        Constant quantities:

            H
            A_ineq

        Online quantities:

            f
            b_ineq
        """

        # --------------------------------------------------
        # Nominal zero operating point
        # --------------------------------------------------

        x0 = np.zeros(
            self.nx,
            dtype=float
        )

        x_ref = np.zeros(
            self.nx,
            dtype=float
        )

        u_prev = np.zeros(
            self.nu,
            dtype=float
        )

        # --------------------------------------------------
        # Initial QP cost
        # --------------------------------------------------

        if self.soft_state_constraints:

            H, f = (
                self.cost.build_soft_qp_matrices(
                    self.Phi,
                    self.Gamma,
                    x0,
                    x_ref,
                    u_prev,
                    self.n_slack
                )
            )

        else:

            H, f = (
                self.cost.build_qp_matrices(
                    self.Phi,
                    self.Gamma,
                    x0,
                    x_ref,
                    u_prev
                )
            )

        # --------------------------------------------------
        # Initial constraints
        # --------------------------------------------------

        if self.soft_state_constraints:

            A_ineq, b_ineq = (
                self.constraints.build_soft_qp_inequalities(
                    self.Phi,
                    self.Gamma,
                    x0,
                    u_prev,
                    self.D
                )
            )

        else:

            A_ineq, b_ineq = (
                self.constraints.build_qp_inequalities(
                    self.Phi,
                    self.Gamma,
                    x0,
                    u_prev,
                    self.D
                )
            )

        # --------------------------------------------------
        # Active optimization dimension
        # --------------------------------------------------

        n_decision = self.nZ

        if H.shape != (
            n_decision,
            n_decision
        ):

            raise ValueError(
                "Invalid Hessian dimensions: "
                f"expected "
                f"({n_decision}, {n_decision}), "
                f"got {H.shape}."
            )

        if f.shape != (
            n_decision,
        ):

            raise ValueError(
                "Invalid gradient dimensions: "
                f"expected ({n_decision},), "
                f"got {f.shape}."
            )

        if A_ineq.shape[1] != n_decision:

            raise ValueError(
                "Invalid constraint matrix dimensions: "
                f"expected {n_decision} columns, "
                f"got {A_ineq.shape[1]}."
            )

        if b_ineq.shape != (
            A_ineq.shape[0],
        ):

            raise ValueError(
                "Invalid constraint vector dimensions."
            )

        # --------------------------------------------------
        # Constraint row scaling for OSQP conditioning
        # --------------------------------------------------

        self.constraint_row_scaling = (
            self._build_constraint_row_scaling(
                A_ineq
            )
        )

        A_ineq = (
            self.constraint_row_scaling[:, None]
            * A_ineq
        )

        b_ineq = (
            self.constraint_row_scaling
            * b_ineq
        )

        # --------------------------------------------------
        # Numerical Hessian symmetry
        # --------------------------------------------------

        H = 0.5 * (
            H + H.T
        )

        # --------------------------------------------------
        # Positive definiteness check
        # --------------------------------------------------

        min_eigenvalue = np.linalg.eigvalsh(
            H
        ).min()

        if min_eigenvalue <= 0.0:

            raise ValueError(
                "QP Hessian must be positive definite."
            )

        # --------------------------------------------------
        # Cache constant matrices
        # --------------------------------------------------

        self.H = H.copy()

        self.A_ineq = (
            A_ineq.copy()
        )

        # --------------------------------------------------
        # OSQP standard form
        #
        # l <= A U <= u
        #
        # We only have:
        #
        # A U <= b
        #
        # therefore:
        #
        # l = -inf
        # u = b
        # --------------------------------------------------

        lower_bound = np.full(
            b_ineq.shape,
            -np.inf,
            dtype=float
        )

        upper_bound = (
            b_ineq.copy()
        )

        self.lower_bound = (
            lower_bound
        )

        # --------------------------------------------------
        # Sparse matrices required by OSQP
        # --------------------------------------------------

        P_sparse = sparse.csc_matrix(
            np.triu(
                self.H
            )
        )

        A_sparse = sparse.csc_matrix(
            self.A_ineq
        )

        # --------------------------------------------------
        # Create OSQP solver
        # --------------------------------------------------

        self.solver = osqp.OSQP()

        # --------------------------------------------------
        # Setup
        # --------------------------------------------------

        self.solver.setup(
            P=P_sparse,
            q=f,
            A=A_sparse,
            l=lower_bound,
            u=upper_bound,

            verbose=False,

            warm_starting=True,

            polishing=False,

            max_iter=int(
                self.max_iterations
            ),

            eps_abs=float(
                self.tolerance
            ),

            eps_rel=float(
                self.tolerance
            )
        )

        self.initialized = True

    # --------------------------------------------------

    def _build_gradient(
        self,
        x0,
        x_ref,
        u_prev
    ):
        """
        Build the online QP gradient.

        f = 2 Gamma.T Qbar e0
            - 2 D.T Sbar b

        where:

            e0 =
                Phi x0 - X_ref

            b =
                [u_prev, 0, ..., 0]

        Parameters
        ----------
        x0 : np.ndarray
            Current state.

        x_ref : np.ndarray
            State reference or stacked trajectory.

        u_prev : np.ndarray
            Previously applied control.

        Returns
        -------
        np.ndarray
            QP gradient f.
        """

        x0 = np.asarray(
            x0,
            dtype=float
        ).reshape(self.nx)

        u_prev = np.asarray(
            u_prev,
            dtype=float
        ).reshape(self.nu)

        X_ref = (
            self.cost.build_reference(
                x_ref
            )
        )

        e0 = (
            self.Phi @ x0
            - X_ref
        )

        b = (
            self.cost.build_delta_offset(
                u_prev
            )
        )

        f = (
            2.0
            * (
                self.Gamma.T
                @ self.Qbar
                @ e0
            )
            -
            2.0
            * (
                self.D.T
                @ self.Sbar
                @ b
            )
        )

        # --------------------------------------------------
        # Augment gradient for soft-QP decision vector
        #
        # f_soft = [f_U; 0]
        # --------------------------------------------------

        if self.soft_state_constraints:

            f_soft = np.zeros(
                self.nZ,
                dtype=float
            )

            f_soft[
                :self.nU
            ] = f

            return f_soft

        return f

    # --------------------------------------------------

    def _build_constraint_upper_bound(
        self,
        x0,
        u_prev
    ):
        """
        Build the online constraint RHS:

            A_ineq U <= b_ineq

        Only b_ineq changes online because
        x0 and u_prev change.

        Parameters
        ----------
        x0 : np.ndarray
            Current state.

        u_prev : np.ndarray
            Previously applied control.

        Returns
        -------
        np.ndarray
            Updated b_ineq.
        """

        x0 = np.asarray(
            x0,
            dtype=float
        ).reshape(self.nx)

        u_prev = np.asarray(
            u_prev,
            dtype=float
        ).reshape(self.nu)

        if self.soft_state_constraints:

            A_current, b_ineq = (
                self.constraints.build_soft_qp_inequalities(
                    self.Phi,
                    self.Gamma,
                    x0,
                    u_prev,
                    self.D
                )
            )

        else:

            A_current, b_ineq = (
                self.constraints.build_qp_inequalities(
                    self.Phi,
                    self.Gamma,
                    x0,
                    u_prev,
                    self.D
                )
            )

        # --------------------------------------------------
        # Apply the same constant row scaling used during
        # OSQP workspace initialization.
        # --------------------------------------------------

        if self.constraint_row_scaling is None:

            raise RuntimeError(
                "Constraint row scaling "
                "has not been initialized."
            )

        A_current_scaled = (
            self.constraint_row_scaling[:, None]
            * A_current
        )

        b_ineq_scaled = (
            self.constraint_row_scaling
            * b_ineq
        )

        # --------------------------------------------------
        # The scaled matrix structure must remain constant.
        # --------------------------------------------------

        if (
            A_current_scaled.shape
            != self.A_ineq.shape
        ):

            raise ValueError(
                "Constraint matrix dimensions changed."
            )

        return b_ineq_scaled

    # --------------------------------------------------

    def solve(
        self,
        x0,
        x_ref,
        u_prev
    ):
        """
        Solve the Physics MPC quadratic program.

        Parameters
        ----------
        x0 : np.ndarray
            Current state vector.

        x_ref : np.ndarray
            State reference or complete
            reference trajectory.

        u_prev : np.ndarray
            Previously applied control input.

        Returns
        -------
        u_opt : np.ndarray
            First optimal control action.

        U_opt : np.ndarray
            Complete optimal control sequence.

        info : dict
            Solver information.
        """

        if not self.initialized:

            raise RuntimeError(
                "Optimizer is not initialized."
            )

        # --------------------------------------------------
        # Validate inputs
        # --------------------------------------------------

        x0 = np.asarray(
            x0,
            dtype=float
        ).reshape(self.nx)

        u_prev = np.asarray(
            u_prev,
            dtype=float
        ).reshape(self.nu)

        # --------------------------------------------------
        # Online gradient
        # --------------------------------------------------

        f = self._build_gradient(
            x0,
            x_ref,
            u_prev
        )

        # --------------------------------------------------
        # Online constraint RHS
        # --------------------------------------------------

        b_ineq = (
            self._build_constraint_upper_bound(
                x0,
                u_prev
            )
        )

        # --------------------------------------------------
        # Update OSQP problem
        #
        # P and A remain unchanged.
        # --------------------------------------------------

        self.solver.update(
            q=f,
            u=b_ineq
        )

        # --------------------------------------------------
        # Solve QP
        # --------------------------------------------------

        result = self.solver.solve(
            raise_error=False
        )

        # --------------------------------------------------
        # Solver status
        # --------------------------------------------------

        status = str(
            result.info.status
        ).lower()

        self.last_status = status

        self.last_objective = (
            result.info.obj_val
        )

        self.last_iterations = (
            result.info.iter
        )

        self.last_solve_time = (
            result.info.solve_time
        )

        self.last_max_iter_hit = False

        # --------------------------------------------------
        # Accept successful statuses
        # --------------------------------------------------

        successful_statuses = {
            "solved",
            "solved inaccurate"
        }

        if status not in successful_statuses:

            # --------------------------------------------------
            # Maximum-iterations acceptance policy
            #
            # A max-iter solution may be reused only if:
            #
            #   1. OSQP returned a primal candidate
            #   2. the candidate is finite
            #   3. the candidate actually satisfies the CURRENT
            #      QP inequalities within a small numerical margin
            #
            # Merely having result.x is NOT sufficient.
            # --------------------------------------------------

            if (
                "maximum iterations reached" in status
                and result.x is not None
            ):

                candidate = np.asarray(
                    result.x,
                    dtype=float
                ).reshape(-1)

                if candidate.size != self.nZ:

                    raise RuntimeError(
                        "OSQP max-iter candidate has invalid "
                        f"size {candidate.size}; expected "
                        f"{self.nZ}."
                    )

                if not np.all(
                    np.isfinite(candidate)
                ):

                    raise RuntimeError(
                        "OSQP max-iter candidate contains "
                        "NaN or Inf."
                    )

                # ----------------------------------------------
                # Direct primal-feasibility verification
                #
                # self.A_ineq and b_ineq use the SAME row-scaled
                # coordinates used by the active OSQP workspace.
                #
                #     A_ineq @ Z <= b_ineq
                #
                # Existing optimizer validation already uses a
                # 10x solver-tolerance feasibility margin.
                # ----------------------------------------------

                constraint_residual = (
                    self.A_ineq @ candidate
                    - b_ineq
                )

                max_constraint_violation = float(
                    np.max(
                        constraint_residual
                    )
                )

                max_iter_feasibility_tol = (
                    10.0
                    * float(
                        self.tolerance
                    )
                )

                if (
                    max_constraint_violation
                    <= max_iter_feasibility_tol
                ):

                    self.last_max_iter_hit = True

                else:

                    raise RuntimeError(
                        "OSQP max-iter candidate rejected: "
                        "constraint violation "
                        f"{max_constraint_violation:.6e} "
                        "exceeds allowed "
                        f"{max_iter_feasibility_tol:.6e}. "
                        f"OSQP primal residual="
                        f"{float(result.info.prim_res):.6e}, "
                        f"dual residual="
                        f"{float(result.info.dual_res):.6e}."
                    )

            else:

                raise RuntimeError(
                    "OSQP failed to solve the MPC QP. "
                    f"Status: {result.info.status}"
                )

        if result.x is None:

            raise RuntimeError(
                "OSQP returned no primal solution."
            )

        # --------------------------------------------------
        # Optimal augmented decision vector
        #
        # Hard QP:
        #     Z_opt = U_opt
        #
        # Soft QP:
        #     Z_opt = [U_opt; epsilon_opt]
        # --------------------------------------------------

        Z_opt = np.asarray(
            result.x,
            dtype=float
        ).reshape(self.nZ)

        # --------------------------------------------------
        # Extract physical MPC control sequence
        # --------------------------------------------------

        U_opt = Z_opt[
            :self.nU
        ].copy()

        # --------------------------------------------------
        # Extract slack variables
        # --------------------------------------------------

        if self.soft_state_constraints:

            epsilon_opt = Z_opt[
                self.nU:
            ].copy()

        else:

            epsilon_opt = np.zeros(
                0,
                dtype=float
            )

        # --------------------------------------------------
        # First receding-horizon control action
        # --------------------------------------------------

        u_opt = U_opt[
            :self.nu
        ].copy()

        # --------------------------------------------------
        # Store solution
        # --------------------------------------------------

        self.last_solution = (
            U_opt.copy()
        )

        self.last_slack_solution = (
            epsilon_opt.copy()
        )

        # --------------------------------------------------
        # Information dictionary
        # --------------------------------------------------

        info = {
            "status": result.info.status,
            "status_val": result.info.status_val,
            "objective": result.info.obj_val,
            "iterations": result.info.iter,
            "solve_time": result.info.solve_time,
            "run_time": result.info.run_time,
            "primal_residual": result.info.prim_res,
            "dual_residual": result.info.dual_res,
            "max_iter_hit": bool(
                self.last_max_iter_hit
            ),
            "slack_max": (
                float(np.max(epsilon_opt))
                if epsilon_opt.size > 0
                else 0.0
            ),
            "slack_norm": float(
                np.linalg.norm(epsilon_opt)
            ),
            "slack_active": bool(
                epsilon_opt.size > 0
                and np.max(epsilon_opt) > 1.0e-8
            )
        }

        return (
            u_opt,
            U_opt,
            info
        )

    # --------------------------------------------------

    def get_last_solution(self):
        """
        Return the most recent optimal
        control sequence.
        """

        return self.last_solution.copy()

    # --------------------------------------------------

    def get_last_slack_solution(self):
        """
        Return the most recent optimal
        soft-constraint slack vector.
        """

        return self.last_slack_solution.copy()

    # --------------------------------------------------

    def reset_warm_start(self):
        """
        Reset the OSQP primal warm-start state.
        """

        if not self.initialized:

            return

        self.solver.warm_start(
            x=np.zeros(
                self.nZ,
                dtype=float
            )
        )

        self.last_solution = np.zeros(
            self.nU,
            dtype=float
        )

        self.last_slack_solution = np.zeros(
            self.n_slack,
            dtype=float
        )