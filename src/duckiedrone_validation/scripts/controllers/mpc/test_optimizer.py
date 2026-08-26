#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_optimizer.py

Validation test for the OSQP-based optimizer
of the Physics-based MPC controller
for the Duckiedrone DD21.

The test validates:

1. MPC parameters
2. Prediction matrices
3. Optimizer initialization
4. Constant QP matrix dimensions
5. Zero-equilibrium solution
6. Zero-equilibrium feasibility
7. Nonzero altitude reference
8. Optimal solution feasibility
9. Objective improvement
10. Predicted trajectory consistency
11. Online QP update
12. Nonzero previous-control handling
13. Delta-U consistency
14. Solver workspace reuse
15. Warm-start reset
16. Infeasible-problem detection

QP problem:

    minimize

        J(U) = 0.5 * U.T @ H @ U
               + f.T @ U

    subject to

        A_ineq @ U <= b_ineq

Author: Abdallah GHOUL 2026
"""

import numpy as np

from .parameters import MPCParameters
from .physics_model import PhysicsModel
from .cost_function import CostFunction
from .constraints import Constraints
from .optimizer import Optimizer


# ============================================================
# Utilities
# ============================================================

def check_feasibility(
    A,
    b,
    U,
    tolerance
):
    """
    Check:

        A @ U <= b

    Parameters
    ----------
    A : np.ndarray
        Inequality matrix.

    b : np.ndarray
        Inequality vector.

    U : np.ndarray
        Control sequence.

    tolerance : float
        Numerical feasibility tolerance.

    Returns
    -------
    feasible : bool
        True if all inequalities are satisfied.

    residual : np.ndarray
        A @ U - b.
    """

    residual = (
        A @ U
        - b
    )

    feasible = np.all(
        residual <= tolerance
    )

    return feasible, residual


def qp_objective(
    H,
    f,
    U
):
    """
    Compute:

        J(U) =
            0.5 * U.T @ H @ U
            + f.T @ U
    """

    return (
        0.5
        * U.T
        @ H
        @ U
        +
        f.T
        @ U
    )


# ============================================================
# Main Test
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" Duckiedrone DD21 - Physics MPC")
    print(" OSQP Optimizer Validation")
    print("=" * 70)

    # --------------------------------------------------------
    # Numerical feasibility tolerance
    #
    # OSQP tolerance is currently 1e-5.
    # Use a slightly relaxed validation tolerance.
    # --------------------------------------------------------

    feasibility_tolerance = 10.0e-5

    # ========================================================
    # 1. MPC Parameters
    # ========================================================

    print()
    print("[1] MPC Parameters")
    print("-" * 70)

    parameters = MPCParameters()

    print("nx =", parameters.nx)
    print("nu =", parameters.nu)
    print("Np =", parameters.Np)
    print("Nc =", parameters.Nc)

    print()
    print(
        "max_iterations =",
        parameters.max_iterations
    )

    print(
        "tolerance      =",
        parameters.tolerance
    )

    assert parameters.nx == 12
    assert parameters.nu == 4
    assert parameters.Np == 20
    assert parameters.Nc == 20

    assert parameters.max_iterations > 0
    assert parameters.tolerance > 0.0

    print()
    print("Parameters validation: PASSED")

    # ========================================================
    # 2. Physics Prediction Model
    # ========================================================

    print()
    print("[2] Physics Prediction Model")
    print("-" * 70)

    model = PhysicsModel(
        parameters
    )

    Phi = model.build_phi()
    Gamma = model.build_gamma()

    print(
        "Phi shape   =",
        Phi.shape
    )

    print(
        "Gamma shape =",
        Gamma.shape
    )

    expected_phi_shape = (
        parameters.Np
        * parameters.nx,
        parameters.nx
    )

    expected_gamma_shape = (
        parameters.Np
        * parameters.nx,
        parameters.Nc
        * parameters.nu
    )

    assert (
        Phi.shape
        == expected_phi_shape
    )

    assert (
        Gamma.shape
        == expected_gamma_shape
    )

    print()
    print(
        "Prediction model validation: PASSED"
    )

    # ========================================================
    # 3. Supporting Objects
    # ========================================================

    print()
    print("[3] Supporting MPC Objects")
    print("-" * 70)

    cost = CostFunction(
        parameters
    )

    constraints = Constraints(
        parameters
    )

    D = cost.build_delta_matrix()

    nU = (
        parameters.Nc
        * parameters.nu
    )

    print(
        "D shape =",
        D.shape
    )

    assert D.shape == (
        nU,
        nU
    )

    print()
    print(
        "Supporting objects validation: PASSED"
    )

    # ========================================================
    # 4. Optimizer Initialization
    # ========================================================

    print()
    print("[4] Optimizer Initialization")
    print("-" * 70)

    optimizer = Optimizer(
        parameters,
        Phi,
        Gamma
    )

    print(
        "Optimizer initialized =",
        optimizer.initialized
    )

    assert optimizer.initialized

    assert optimizer.solver is not None

    print(
        "H shape =",
        optimizer.H.shape
    )

    print(
        "A_ineq shape =",
        optimizer.A_ineq.shape
    )

    # --------------------------------------------------------
    # Active optimization dimensions
    # --------------------------------------------------------

    if optimizer.soft_state_constraints:

        expected_n_slack = (
            3
            * parameters.Np
        )

        expected_n_decision = (
            nU
            + expected_n_slack
        )

        assert optimizer.n_slack == (
            expected_n_slack
        )

        assert optimizer.nZ == (
            expected_n_decision
        )

        assert optimizer.H.shape == (
            expected_n_decision,
            expected_n_decision
        )

        assert (
            optimizer.A_ineq.shape[1]
            == expected_n_decision
        )

        # ----------------------------------------------------
        # Constraint count - Soft QP
        #
        # Hard input:
        #     2 * nU
        #
        # Hard Delta-U:
        #     2 * nU
        #
        # Hard rotor feasibility:
        #     2 * nU
        #
        # Soft state:
        #     6 * Np
        #
        # Slack non-negativity:
        #     3 * Np
        #
        # For Nc = 20, nu = 4:
        #
        #     nU      = 80
        #     n_slack = 60
        #
        #     160 + 160 + 160
        #     + 120 + 60
        #     = 660
        # ----------------------------------------------------

        expected_constraint_count = (
            2 * nU
            + 2 * nU
            + 2 * nU
            + 6 * parameters.Np
            + expected_n_slack
        )

        expected_A_ineq_shape = (
            expected_constraint_count,
            expected_n_decision
        )

    else:

        assert optimizer.n_slack == 0
        assert optimizer.nZ == nU

        assert optimizer.H.shape == (
            nU,
            nU
        )

        assert (
            optimizer.A_ineq.shape[1]
            == nU
        )

        # ----------------------------------------------------
        # Constraint count - Hard QP
        #
        # Input:
        #     2 * nU
        #
        # Delta-U:
        #     2 * nU
        #
        # Rotor feasibility:
        #     2 * nU
        #
        # State:
        #     6 * Np
        #
        #     160 + 160 + 160 + 120
        #     = 600
        # ----------------------------------------------------

        expected_constraint_count = (
            2 * nU
            + 2 * nU
            + 2 * nU
            + 6 * parameters.Np
        )

        expected_A_ineq_shape = (
            expected_constraint_count,
            nU
        )

    print(
        "soft_state_constraints =",
        optimizer.soft_state_constraints
    )

    print(
        "nU =",
        optimizer.nU
    )

    print(
        "n_slack =",
        optimizer.n_slack
    )

    print(
        "nZ =",
        optimizer.nZ
    )

    print(
        "Expected constraints =",
        expected_constraint_count
    )

    assert (
        optimizer.A_ineq.shape
        == expected_A_ineq_shape
    )

    # ========================================================
    # 5. Constant Hessian Validation
    # ========================================================

    print()
    print("[5] Constant Hessian Validation")
    print("-" * 70)

    x_zero = np.zeros(
        parameters.nx
    )

    x_ref_zero = np.zeros(
        parameters.nx
    )

    u_prev_zero = np.zeros(
        parameters.nu
    )

    if optimizer.soft_state_constraints:

        H_expected, f_zero_expected = (
            cost.build_soft_qp_matrices(
                Phi,
                Gamma,
                x_zero,
                x_ref_zero,
                u_prev_zero,
                optimizer.n_slack
            )
        )

    else:

        H_expected, f_zero_expected = (
            cost.build_qp_matrices(
                Phi,
                Gamma,
                x_zero,
                x_ref_zero,
                u_prev_zero
            )
        )
    

    H_correct = np.allclose(
        optimizer.H,
        H_expected
    )

    print(
        "Optimizer H matches CostFunction H =",
        H_correct
    )

    assert H_correct

    # --------------------------------------------------------
    # Soft-QP block structure validation
    # --------------------------------------------------------

    if optimizer.soft_state_constraints:

        H_U = optimizer.H[
            :optimizer.nU,
            :optimizer.nU
        ]

        H_U_eps = optimizer.H[
            :optimizer.nU,
            optimizer.nU:
        ]

        H_eps_U = optimizer.H[
            optimizer.nU:,
            :optimizer.nU
        ]

        H_eps = optimizer.H[
            optimizer.nU:,
            optimizer.nU:
        ]

        H_U_expected, _ = (
            cost.build_qp_matrices(
                Phi,
                Gamma,
                x_zero,
                x_ref_zero,
                u_prev_zero
            )
        )

        H_eps_expected = (
            2.0
            * float(parameters.slack_weight)
            * np.eye(
                optimizer.n_slack
            )
        )

        control_block_correct = np.allclose(
            H_U,
            H_U_expected
        )

        upper_cross_zero = np.allclose(
            H_U_eps,
            0.0
        )

        lower_cross_zero = np.allclose(
            H_eps_U,
            0.0
        )

        slack_block_correct = np.allclose(
            H_eps,
            H_eps_expected
        )

        print(
            "Control Hessian block correct =",
            control_block_correct
        )

        print(
            "Upper cross block zero =",
            upper_cross_zero
        )

        print(
            "Lower cross block zero =",
            lower_cross_zero
        )

        print(
            "Slack Hessian block correct =",
            slack_block_correct
        )

        assert control_block_correct
        assert upper_cross_zero
        assert lower_cross_zero
        assert slack_block_correct

    H_symmetric = np.allclose(
        optimizer.H,
        optimizer.H.T
    )

    print(
        "H symmetric =",
        H_symmetric
    )

    assert H_symmetric

    eigvals = np.linalg.eigvalsh(
        optimizer.H
    )

    min_eig = eigvals.min()
    max_eig = eigvals.max()

    print(
        "Minimum eigenvalue =",
        min_eig
    )

    print(
        "Maximum eigenvalue =",
        max_eig
    )

    assert min_eig > 0.0

    print()
    print(
        "Constant Hessian validation: PASSED"
    )

    # ========================================================
    # 6. Zero-Equilibrium Solution
    # ========================================================

    print()
    print("[6] Zero-Equilibrium Solution")
    print("-" * 70)

    u_zero_opt, U_zero_opt, info_zero = (
        optimizer.solve(
            x_zero,
            x_ref_zero,
            u_prev_zero
        )
    )

    print(
        "Solver status =",
        info_zero["status"]
    )

    print(
        "Iterations =",
        info_zero["iterations"]
    )

    print(
        "Objective =",
        info_zero["objective"]
    )

    print(
        "Solve time =",
        info_zero["solve_time"]
    )

    print(
        "Primal residual =",
        info_zero["primal_residual"]
    )

    print(
        "Dual residual =",
        info_zero["dual_residual"]
    )

    print()

    print(
        "u_opt =",
        u_zero_opt
    )

    print(
        "||U_opt|| =",
        np.linalg.norm(
            U_zero_opt
        )
    )

    assert u_zero_opt.shape == (
        parameters.nu,
    )

    assert U_zero_opt.shape == (
        nU,
    )

    # For:
    #
    # x0 = 0
    # x_ref = 0
    # u_prev = 0
    #
    # theoretical optimum:
    #
    # U* = 0

    zero_solution_correct = np.allclose(
        U_zero_opt,
        0.0,
        atol=1e-7
    )

    print(
        "Zero solution correct =",
        zero_solution_correct
    )

    assert zero_solution_correct

    print()
    print(
        "Zero-equilibrium solution: PASSED"
    )

    # ========================================================
    # 7. Zero-Equilibrium Feasibility
    # ========================================================

    print()
    print("[7] Zero-Equilibrium Feasibility")
    print("-" * 70)

    A_zero, b_zero = (
        constraints.build_qp_inequalities(
            Phi,
            Gamma,
            x_zero,
            u_prev_zero,
            D
        )
    )

    zero_feasible, residual_zero = (
        check_feasibility(
            A_zero,
            b_zero,
            U_zero_opt,
            feasibility_tolerance
        )
    )

    max_zero_residual = np.max(
        residual_zero
    )

    print(
        "Solution feasible =",
        zero_feasible
    )

    print(
        "Maximum residual =",
        max_zero_residual
    )

    assert zero_feasible

    print()
    print(
        "Zero-equilibrium feasibility: PASSED"
    )

    # ========================================================
    # 8. Altitude Reference
    # ========================================================

    print()
    print("[8] Altitude Reference Test")
    print("-" * 70)

    x_ref_altitude = np.zeros(
        parameters.nx
    )

    # State ordering:
    #
    # [x, y, z,
    #  phi, theta, psi,
    #  vx, vy, vz,
    #  p, q, r]

    x_ref_altitude[2] = 1.0

    u_alt_opt, U_alt_opt, info_alt = (
        optimizer.solve(
            x_zero,
            x_ref_altitude,
            u_prev_zero
        )
    )

    print(
        "Solver status =",
        info_alt["status"]
    )

    print(
        "Iterations =",
        info_alt["iterations"]
    )

    print(
        "Objective =",
        info_alt["objective"]
    )

    print(
        "Solve time =",
        info_alt["solve_time"]
    )

    print()

    print(
        "First optimal control =",
        u_alt_opt
    )

    print(
        "||U_opt|| =",
        np.linalg.norm(
            U_alt_opt
        )
    )

    assert U_alt_opt.shape == (
        nU,
    )

    assert np.all(
        np.isfinite(
            U_alt_opt
        )
    )

    nonzero_altitude_control = (
        np.linalg.norm(
            U_alt_opt
        ) > 1e-9
    )

    print(
        "Nonzero control generated =",
        nonzero_altitude_control
    )

    assert nonzero_altitude_control

    print()
    print(
        "Altitude reference solution: PASSED"
    )

    # ========================================================
    # 9. Altitude Solution Feasibility
    # ========================================================

    print()
    print("[9] Altitude Solution Feasibility")
    print("-" * 70)

    A_alt, b_alt = (
        constraints.build_qp_inequalities(
            Phi,
            Gamma,
            x_zero,
            u_prev_zero,
            D
        )
    )

    altitude_feasible, residual_alt = (
        check_feasibility(
            A_alt,
            b_alt,
            U_alt_opt,
            feasibility_tolerance
        )
    )

    max_alt_residual = np.max(
        residual_alt
    )

    print(
        "Altitude solution feasible =",
        altitude_feasible
    )

    print(
        "Maximum residual =",
        max_alt_residual
    )

    assert altitude_feasible

    print()
    print(
        "Altitude solution feasibility: PASSED"
    )

    # ========================================================
    # 10. Objective Improvement
    # ========================================================

    print()
    print("[10] Objective Improvement")
    print("-" * 70)

    H_alt, f_alt = (
        cost.build_qp_matrices(
            Phi,
            Gamma,
            x_zero,
            x_ref_altitude,
            u_prev_zero
        )
    )

    J_zero_control = qp_objective(
        H_alt,
        f_alt,
        np.zeros(
            nU
        )
    )

    J_optimal = qp_objective(
        H_alt,
        f_alt,
        U_alt_opt
    )

    print(
        "J(U = 0) =",
        J_zero_control
    )

    print(
        "J(U_opt) =",
        J_optimal
    )

    objective_improved = (
        J_optimal
        <= J_zero_control
        + 1e-7
    )

    print(
        "Optimal objective improved =",
        objective_improved
    )

    assert objective_improved

    print()
    print(
        "Objective improvement: PASSED"
    )

    # ========================================================
    # 11. Predicted Trajectory
    # ========================================================

    print()
    print("[11] Predicted Trajectory")
    print("-" * 70)

    X_alt = (
        Phi @ x_zero
        + Gamma @ U_alt_opt
    )

    assert X_alt.shape == (
        parameters.Np
        * parameters.nx,
    )

    X_alt_steps = X_alt.reshape(
        parameters.Np,
        parameters.nx
    )

    z_prediction = (
        X_alt_steps[:, 2]
    )

    print(
        "Initial predicted z =",
        z_prediction[0]
    )

    print(
        "Final predicted z =",
        z_prediction[-1]
    )

    print(
        "Maximum predicted z =",
        np.max(
            z_prediction
        )
    )

    predicted_altitude_finite = np.all(
        np.isfinite(
            z_prediction
        )
    )

    print(
        "Predicted altitude finite =",
        predicted_altitude_finite
    )

    assert predicted_altitude_finite

    # With a positive altitude reference,
    # the predicted altitude should move
    # in the positive direction.

    altitude_moves_positive = (
        z_prediction[-1] > 0.0
    )

    print(
        "Predicted altitude moves positive =",
        altitude_moves_positive
    )

    assert altitude_moves_positive

    print()
    print(
        "Predicted trajectory validation: PASSED"
    )

    # ========================================================
    # 12. Solver Workspace Reuse
    # ========================================================

    print()
    print("[12] Solver Workspace Reuse")
    print("-" * 70)

    solver_id_before = id(
        optimizer.solver
    )

    # Change reference
    x_ref_altitude_2 = np.zeros(
        parameters.nx
    )

    x_ref_altitude_2[2] = 0.5

    u_alt2_opt, U_alt2_opt, info_alt2 = (
        optimizer.solve(
            x_zero,
            x_ref_altitude_2,
            u_prev_zero
        )
    )

    solver_id_after = id(
        optimizer.solver
    )

    workspace_reused = (
        solver_id_before
        == solver_id_after
    )

    print(
        "Solver workspace reused =",
        workspace_reused
    )

    print(
        "Second status =",
        info_alt2["status"]
    )

    print(
        "Second iterations =",
        info_alt2["iterations"]
    )

    print(
        "Second solve time =",
        info_alt2["solve_time"]
    )

    assert workspace_reused

    assert np.all(
        np.isfinite(
            U_alt2_opt
        )
    )

    print()
    print(
        "Solver workspace reuse: PASSED"
    )

    # ========================================================
    # 13. Nonzero Previous Control
    # ========================================================

    print()
    print("[13] Nonzero Previous Control")
    print("-" * 70)

    u_prev_nonzero = np.array(
        [
            0.30,
            0.00,
            0.00,
            0.00
        ],
        dtype=float
    )

    x_ref_nonzero_prev = np.zeros(
        parameters.nx
    )

    x_ref_nonzero_prev[2] = 0.5

    u_prev_opt, U_prev_opt, info_prev = (
        optimizer.solve(
            x_zero,
            x_ref_nonzero_prev,
            u_prev_nonzero
        )
    )

    print(
        "Solver status =",
        info_prev["status"]
    )

    print(
        "First optimal control =",
        u_prev_opt
    )

    assert np.all(
        np.isfinite(
            U_prev_opt
        )
    )

    # --------------------------------------------------------
    # Compute Delta_U
    # --------------------------------------------------------

    b_offset = (
        cost.build_delta_offset(
            u_prev_nonzero
        )
    )

    Delta_U = (
        D @ U_prev_opt
        - b_offset
    )

    Delta_steps = Delta_U.reshape(
        parameters.Nc,
        parameters.nu
    )

    print(
        "First Delta_u =",
        Delta_steps[0]
    )

    print(
        "Maximum |Delta_u| =",
        np.max(
            np.abs(
                Delta_U
            )
        )
    )

    # Current implementation:
    #
    # symmetric du_max
    # on all channels

    delta_constraints_respected = np.all(
        np.abs(
            Delta_U
        )
        <= (
            parameters.du_max
            + feasibility_tolerance
        )
    )

    print(
        "Delta-U limits respected =",
        delta_constraints_respected
    )

    assert delta_constraints_respected

    print()
    print(
        "Nonzero previous control: PASSED"
    )

    # ========================================================
    # 14. Nonzero Previous-Control Feasibility
    # ========================================================

    print()
    print("[14] Nonzero Previous-Control Feasibility")
    print("-" * 70)

    A_prev, b_prev = (
        constraints.build_qp_inequalities(
            Phi,
            Gamma,
            x_zero,
            u_prev_nonzero,
            D
        )
    )

    prev_feasible, residual_prev = (
        check_feasibility(
            A_prev,
            b_prev,
            U_prev_opt,
            feasibility_tolerance
        )
    )

    print(
        "Solution feasible =",
        prev_feasible
    )

    print(
        "Maximum residual =",
        np.max(
            residual_prev
        )
    )

    assert prev_feasible

    print()
    print(
        "Previous-control feasibility: PASSED"
    )

    # ========================================================
    # 15. Online Gradient Verification
    # ========================================================

    print()
    print("[15] Online Gradient Verification")
    print("-" * 70)

    f_optimizer = (
        optimizer._build_gradient(
            x_zero,
            x_ref_nonzero_prev,
            u_prev_nonzero
        )
    )

    if optimizer.soft_state_constraints:

        H_direct, f_direct = (
            cost.build_soft_qp_matrices(
                Phi,
                Gamma,
                x_zero,
                x_ref_nonzero_prev,
                u_prev_nonzero,
                optimizer.n_slack
            )
        )

    else:

        H_direct, f_direct = (
            cost.build_qp_matrices(
                Phi,
                Gamma,
                x_zero,
                x_ref_nonzero_prev,
                u_prev_nonzero
            )
        )

    gradient_correct = np.allclose(
        f_optimizer,
        f_direct
    )

    print(
        "Online gradient correct =",
        gradient_correct
    )

    assert gradient_correct

    # --------------------------------------------------------
    # Soft-QP gradient structure
    # --------------------------------------------------------

    if optimizer.soft_state_constraints:

        f_control = f_optimizer[
            :optimizer.nU
        ]

        f_slack = f_optimizer[
            optimizer.nU:
        ]

        _, f_control_expected = (
            cost.build_qp_matrices(
                Phi,
                Gamma,
                x_zero,
                x_ref_nonzero_prev,
                u_prev_nonzero
            )
        )

        control_gradient_correct = np.allclose(
            f_control,
            f_control_expected
        )

        slack_gradient_zero = np.allclose(
            f_slack,
            0.0
        )

        print(
            "Control gradient block correct =",
            control_gradient_correct
        )

        print(
            "Slack gradient zero =",
            slack_gradient_zero
        )

        assert control_gradient_correct
        assert slack_gradient_zero

    # H remains constant
    H_still_constant = np.allclose(
        optimizer.H,
        H_direct
    )

    print(
        "H remains constant =",
        H_still_constant
    )

    assert H_still_constant

    print()
    print(
        "Online gradient verification: PASSED"
    )

    # ========================================================
    # 16. Online Constraint RHS Verification
    # ========================================================

    print()
    print("[16] Online Constraint RHS Verification")
    print("-" * 70)

    b_optimizer = (
        optimizer._build_constraint_upper_bound(
            x_zero,
            u_prev_nonzero
        )
    )

    if optimizer.soft_state_constraints:

        A_direct, b_direct = (
            constraints.build_soft_qp_inequalities(
                Phi,
                Gamma,
                x_zero,
                u_prev_nonzero,
                D
            )
        )

    else:

        A_direct, b_direct = (
            constraints.build_qp_inequalities(
                Phi,
                Gamma,
                x_zero,
                u_prev_nonzero,
                D
            )
        )

    # --------------------------------------------------------
    # Apply the same row scaling used by the optimizer.
    # --------------------------------------------------------

    A_direct_scaled = (
        optimizer.constraint_row_scaling[:, None]
        * A_direct
    )

    b_direct_scaled = (
        optimizer.constraint_row_scaling
        * b_direct
    )

    print(
        "b_optimizer shape =",
        b_optimizer.shape
    )

    print(
        "b_direct_scaled shape =",
        b_direct_scaled.shape
    )

    print(
        "A_direct shape =",
        A_direct.shape
    )

    assert (
        b_optimizer.shape
        == b_direct_scaled.shape
    )

    assert (
        A_direct.shape
        == optimizer.A_ineq.shape
    )


    rhs_correct = np.allclose(
        b_optimizer,
        b_direct_scaled
    )

    print(
        "Online constraint RHS correct =",
        rhs_correct
    )

    assert rhs_correct

    A_constant = np.allclose(
        optimizer.A_ineq,
        A_direct_scaled
    )

    print(
        "Constraint matrix A remains constant =",
        A_constant
    )

    rotor_start = (
        4
        * optimizer.nU
    )

    rotor_end = (
        6
        * optimizer.nU
    )

    rotor_scaled_max = np.max(
        np.abs(
            optimizer.A_ineq[
                rotor_start:
                rotor_end
            ]
        )
    )

    print(
        "Scaled rotor max |A| =",
        rotor_scaled_max
    )

    assert np.isclose(
        rotor_scaled_max,
        1.0
    )

    assert A_constant

    print()
    print(
        "Online constraint verification: PASSED"
    )

    # ========================================================
    # 17. Last Solution Storage
    # ========================================================

    print()
    print("[17] Last Solution Storage")
    print("-" * 70)

    last_solution = (
        optimizer.get_last_solution()
    )

    last_solution_correct = np.allclose(
        last_solution,
        U_prev_opt
    )

    print(
        "Last solution stored correctly =",
        last_solution_correct
    )

    assert last_solution_correct

    print()
    print(
        "Last solution storage: PASSED"
    )

    # ========================================================
    # 18. Warm-Start Reset
    # ========================================================

    print()
    print("[18] Warm-Start Reset")
    print("-" * 70)

    optimizer.reset_warm_start()

    reset_solution = (
        optimizer.get_last_solution()
    )

    reset_correct = np.allclose(
        reset_solution,
        0.0
    )

    print(
        "Stored solution reset to zero =",
        reset_correct
    )

    assert reset_correct

    # Solver must remain initialized
    assert optimizer.initialized

    print(
        "Optimizer still initialized =",
        optimizer.initialized
    )

    print()
    print(
        "Warm-start reset: PASSED"
    )

    # ========================================================
    # 19. Solve After Warm-Start Reset
    # ========================================================

    print()
    print("[19] Solve After Warm-Start Reset")
    print("-" * 70)

    u_reset_opt, U_reset_opt, info_reset = (
        optimizer.solve(
            x_zero,
            x_ref_altitude,
            u_prev_zero
        )
    )

    print(
        "Status after reset =",
        info_reset["status"]
    )

    print(
        "Iterations after reset =",
        info_reset["iterations"]
    )

    print(
        "First control after reset =",
        u_reset_opt
    )

    assert np.all(
        np.isfinite(
            U_reset_opt
        )
    )

    print()
    print(
        "Solve after warm-start reset: PASSED"
    )

    # ========================================================
    # 20. Infeasible Problem Detection
    # ========================================================

    print()
    print("[20] Infeasible Problem Detection")
    print("-" * 70)

    # --------------------------------------------------------
    # Put the initial altitude clearly above z_max.
    #
    # Because z dynamics have relative degree > 0,
    # the first predicted altitude cannot be corrected
    # instantaneously by the current control input.
    #
    # This should make the constrained QP infeasible.
    # --------------------------------------------------------

    x_infeasible = np.zeros(
        parameters.nx
    )

    x_infeasible[2] = (
        parameters.z_max
        + 1.0
    )

    infeasible_detected = False
    infeasible_message = None

    try:

        optimizer.solve(
            x_infeasible,
            x_ref_zero,
            u_prev_zero
        )

    except RuntimeError as error:

        infeasible_detected = True
        infeasible_message = str(
            error
        )

    print(
        "Infeasible case detected =",
        infeasible_detected
    )

    if infeasible_message is not None:

        print(
            "Solver message =",
            infeasible_message
        )

    assert infeasible_detected

    print()
    print(
        "Infeasible-problem detection: PASSED"
    )

    # ========================================================
    # 21. Final Results
    # ========================================================

    print()
    print("=" * 70)
    print(" OPTIMIZER TEST RESULTS")
    print("=" * 70)

    print(
        "Phi     :",
        Phi.shape
    )

    print(
        "Gamma   :",
        Gamma.shape
    )

    print(
        "H       :",
        optimizer.H.shape
    )

    print(
        "A_ineq  :",
        optimizer.A_ineq.shape
    )

    print(
        "U size  :",
        nU
    )

    print()

    print(
        "H minimum eigenvalue =",
        min_eig
    )

    print(
        "H maximum eigenvalue =",
        max_eig
    )

    print()

    print(
        "Zero solution norm =",
        np.linalg.norm(
            U_zero_opt
        )
    )

    print(
        "Altitude solution norm =",
        np.linalg.norm(
            U_alt_opt
        )
    )

    print()

    print(
        "Altitude first control =",
        u_alt_opt
    )

    print(
        "Altitude final predicted z =",
        z_prediction[-1]
    )

    print()

    print(
        "Zero-case iterations =",
        info_zero["iterations"]
    )

    print(
        "Altitude iterations =",
        info_alt["iterations"]
    )

    print(
        "Second solve iterations =",
        info_alt2["iterations"]
    )

    print()

    print(
        "Zero-case solve time =",
        info_zero["solve_time"]
    )

    print(
        "Altitude solve time =",
        info_alt["solve_time"]
    )

    print(
        "Second solve time =",
        info_alt2["solve_time"]
    )

    print()

    print(
        "Workspace reused =",
        workspace_reused
    )

    print(
        "Infeasible case detected =",
        infeasible_detected
    )

    print()
    print(
        "ALL OPTIMIZER TESTS PASSED"
    )

    print("=" * 70)
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()