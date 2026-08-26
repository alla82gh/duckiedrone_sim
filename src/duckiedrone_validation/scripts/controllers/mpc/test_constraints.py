#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_constraints.py

Validation test for the linear constraints
of the Physics-based MPC controller
for the Duckiedrone DD21.

The test validates:

1. MPC dimensions
2. Input bounds
3. Control increment bounds
4. State bounds
5. Input inequalities
6. Delta-U inequalities
7. State inequalities
8. Complete QP inequality system
9. Numerical feasibility
10. Numerical violation detection
11. Direct analytical consistency

QP constraints:

    A_ineq @ U <= b_ineq

Prediction model:

    X = Phi @ x0 + Gamma @ U

Control increments:

    Delta_U = D @ U - b

Author: Abdallah GHOUL 2026
"""

import numpy as np

from .parameters import MPCParameters
from .physics_model import PhysicsModel
from .cost_function import CostFunction
from .constraints import Constraints


# ============================================================
# Utility
# ============================================================

def inequality_satisfied(
    A,
    b,
    U,
    tolerance=1e-10
):
    """
    Check whether:

        A @ U <= b

    within a numerical tolerance.
    """

    residual = A @ U - b

    satisfied = np.all(
        residual <= tolerance
    )

    return satisfied, residual


# ============================================================
# Main Test
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" Duckiedrone DD21 - Physics MPC")
    print(" Constraints Validation")
    print("=" * 70)

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
    print("phi_max    =", parameters.phi_max)
    print("theta_max  =", parameters.theta_max)
    print("z_min      =", parameters.z_min)
    print("z_max      =", parameters.z_max)
    print("thrust_min =", parameters.thrust_min)
    print("thrust_max =", parameters.thrust_max)
    print("torque_max =", parameters.torque_max)
    print("du_max     =", parameters.du_max)

    assert parameters.nx == 12
    assert parameters.nu == 4
    assert parameters.Np == 20
    assert parameters.Nc == 20

    print()
    print("Parameters validation: PASSED")

    # ========================================================
    # 2. Prediction Model
    # ========================================================

    print()
    print("[2] Physics Prediction Model")
    print("-" * 70)

    model = PhysicsModel(parameters)

    Phi = model.build_phi()
    Gamma = model.build_gamma()

    print("Phi shape   =", Phi.shape)
    print("Gamma shape =", Gamma.shape)

    assert Phi.shape == (
        parameters.Np * parameters.nx,
        parameters.nx
    )

    assert Gamma.shape == (
        parameters.Np * parameters.nx,
        parameters.Nc * parameters.nu
    )

    print()
    print("Prediction model validation: PASSED")

    # ========================================================
    # 3. Constraint Objects
    # ========================================================

    print()
    print("[3] Constraint Objects")
    print("-" * 70)

    cost = CostFunction(parameters)

    constraints = Constraints(
        parameters
    )

    D = cost.build_delta_matrix()

    print("CostFunction created.")
    print("Constraints created.")
    print("D shape =", D.shape)

    assert D.shape == (
        parameters.Nc * parameters.nu,
        parameters.Nc * parameters.nu
    )

    print()
    print("Constraint objects validation: PASSED")

    # ========================================================
    # 4. Input Bounds
    # ========================================================

    print()
    print("[4] Input Bounds Validation")
    print("-" * 70)

    U_min, U_max = (
        constraints.build_input_bounds()
    )

    expected_u_shape = (
        parameters.Nc * parameters.nu,
    )

    print("U_min shape =", U_min.shape)
    print("U_max shape =", U_max.shape)

    assert U_min.shape == expected_u_shape
    assert U_max.shape == expected_u_shape

    hover_thrust = (
        constraints.vehicle.mass
        * constraints.vehicle.gravity
    )

    expected_single_min = np.array(
        [
            parameters.thrust_min - hover_thrust,
            -parameters.torque_max,
            -parameters.torque_max,
            -parameters.torque_max
        ],
        dtype=float
    )

    expected_single_max = np.array(
        [
            parameters.thrust_max - hover_thrust,
            parameters.torque_max,
            parameters.torque_max,
            parameters.torque_max
        ],
        dtype=float
    )

    first_u_min_correct = np.allclose(
        U_min[:parameters.nu],
        expected_single_min
    )

    first_u_max_correct = np.allclose(
        U_max[:parameters.nu],
        expected_single_max
    )

    last_u_min_correct = np.allclose(
        U_min[-parameters.nu:],
        expected_single_min
    )

    last_u_max_correct = np.allclose(
        U_max[-parameters.nu:],
        expected_single_max
    )

    print(
        "First U_min block correct =",
        first_u_min_correct
    )

    print(
        "First U_max block correct =",
        first_u_max_correct
    )

    print(
        "Last U_min block correct =",
        last_u_min_correct
    )

    print(
        "Last U_max block correct =",
        last_u_max_correct
    )

    assert first_u_min_correct
    assert first_u_max_correct
    assert last_u_min_correct
    assert last_u_max_correct

    assert np.all(
        U_min < U_max
    )

    print()
    print("Input bounds validation: PASSED")

    # ========================================================
    # 5. Delta-U Bounds
    # ========================================================

    print()
    print("[5] Delta-U Bounds Validation")
    print("-" * 70)

    Delta_U_min, Delta_U_max = (
        constraints.build_delta_bounds()
    )

    expected_delta_shape = (
        parameters.Nc * parameters.nu,
    )

    print(
        "Delta_U_min shape =",
        Delta_U_min.shape
    )

    print(
        "Delta_U_max shape =",
        Delta_U_max.shape
    )

    assert (
        Delta_U_min.shape
        == expected_delta_shape
    )

    assert (
        Delta_U_max.shape
        == expected_delta_shape
    )

    expected_delta_min = (
        -parameters.du_max
        * np.ones(
            expected_delta_shape[0]
        )
    )

    expected_delta_max = (
        parameters.du_max
        * np.ones(
            expected_delta_shape[0]
        )
    )

    delta_min_correct = np.allclose(
        Delta_U_min,
        expected_delta_min
    )

    delta_max_correct = np.allclose(
        Delta_U_max,
        expected_delta_max
    )

    print(
        "Delta_U_min correct =",
        delta_min_correct
    )

    print(
        "Delta_U_max correct =",
        delta_max_correct
    )

    assert delta_min_correct
    assert delta_max_correct

    print()
    print("Delta-U bounds validation: PASSED")

    # ========================================================
    # 6. State Bounds
    # ========================================================

    print()
    print("[6] State Bounds Validation")
    print("-" * 70)

    X_min, X_max = (
        constraints.build_state_bounds()
    )

    expected_state_shape = (
        parameters.Np * parameters.nx,
    )

    print("X_min shape =", X_min.shape)
    print("X_max shape =", X_max.shape)

    assert X_min.shape == expected_state_shape
    assert X_max.shape == expected_state_shape

    # --------------------------------------------------------
    # State ordering:
    #
    # 0 x
    # 1 y
    # 2 z
    # 3 phi
    # 4 theta
    # 5 psi
    # 6 vx
    # 7 vy
    # 8 vz
    # 9 p
    # 10 q
    # 11 r
    # --------------------------------------------------------

    first_x_min = X_min[
        :parameters.nx
    ]

    first_x_max = X_max[
        :parameters.nx
    ]

    print()
    print(
        "z bounds =",
        first_x_min[2],
        first_x_max[2]
    )

    print(
        "phi bounds =",
        first_x_min[3],
        first_x_max[3]
    )

    print(
        "theta bounds =",
        first_x_min[4],
        first_x_max[4]
    )

    assert np.isclose(
        first_x_min[2],
        parameters.z_min
    )

    assert np.isclose(
        first_x_max[2],
        parameters.z_max
    )

    assert np.isclose(
        first_x_min[3],
        -parameters.phi_max
    )

    assert np.isclose(
        first_x_max[3],
        parameters.phi_max
    )

    assert np.isclose(
        first_x_min[4],
        -parameters.theta_max
    )

    assert np.isclose(
        first_x_max[4],
        parameters.theta_max
    )

    # Exactly 3 finite lower bounds per stage:
    #
    # z
    # phi
    # theta

    finite_lower_count = np.sum(
        np.isfinite(X_min)
    )

    finite_upper_count = np.sum(
        np.isfinite(X_max)
    )

    expected_finite_count = (
        3 * parameters.Np
    )

    print()
    print(
        "Finite lower bounds =",
        finite_lower_count
    )

    print(
        "Finite upper bounds =",
        finite_upper_count
    )

    assert (
        finite_lower_count
        == expected_finite_count
    )

    assert (
        finite_upper_count
        == expected_finite_count
    )

    print()
    print("State bounds validation: PASSED")

    # ========================================================
    # 7. Input Inequalities
    # ========================================================

    print()
    print("[7] Input Inequalities")
    print("-" * 70)

    A_u, b_u = (
        constraints.build_input_inequalities()
    )

    nU = (
        parameters.Nc
        * parameters.nu
    )

    expected_A_u_shape = (
        2 * nU,
        nU
    )

    expected_b_u_shape = (
        2 * nU,
    )

    print("A_u shape =", A_u.shape)
    print("b_u shape =", b_u.shape)

    assert A_u.shape == expected_A_u_shape
    assert b_u.shape == expected_b_u_shape

    I = np.eye(
        nU
    )

    assert np.allclose(
        A_u[:nU],
        I
    )

    assert np.allclose(
        A_u[nU:],
        -I
    )

    assert np.allclose(
        b_u[:nU],
        U_max
    )

    assert np.allclose(
        b_u[nU:],
        -U_min
    )

    print(
        "Input inequality structure correct = True"
    )

    print()
    print("Input inequalities validation: PASSED")

    # ========================================================
    # 8. Delta-U Inequalities
    # ========================================================

    print()
    print("[8] Delta-U Inequalities")
    print("-" * 70)

    u_prev_test = np.array(
        [
            0.50,
            -0.20,
            0.10,
            0.05
        ],
        dtype=float
    )

    A_du, b_du = (
        constraints.build_delta_inequalities(
            D,
            u_prev_test
        )
    )

    expected_A_du_shape = (
        2 * nU,
        nU
    )

    expected_b_du_shape = (
        2 * nU,
    )

    print("A_du shape =", A_du.shape)
    print("b_du shape =", b_du.shape)

    assert (
        A_du.shape
        == expected_A_du_shape
    )

    assert (
        b_du.shape
        == expected_b_du_shape
    )

    # --------------------------------------------------------
    # Check A structure
    # --------------------------------------------------------

    assert np.allclose(
        A_du[:nU],
        D
    )

    assert np.allclose(
        A_du[nU:],
        -D
    )

    # --------------------------------------------------------
    # Build expected b vector
    # --------------------------------------------------------

    b_offset = (
        cost.build_delta_offset(
            u_prev_test
        )
    )

    expected_b_du = np.concatenate(
        (
            Delta_U_max
            + b_offset,

            -Delta_U_min
            - b_offset
        )
    )

    delta_rhs_correct = np.allclose(
        b_du,
        expected_b_du
    )

    print(
        "Delta inequality RHS correct =",
        delta_rhs_correct
    )

    assert delta_rhs_correct

    print()
    print("Delta-U inequalities validation: PASSED")

    # ========================================================
    # 9. Numerical Delta-U Feasibility
    # ========================================================

    print()
    print("[9] Delta-U Numerical Feasibility")
    print("-" * 70)

    # Keep future controls equal to previous control:
    #
    # u0 = u_prev
    # u1 = u_prev
    # ...
    #
    # Therefore:
    #
    # Delta_U = 0

    U_hold = np.tile(
        u_prev_test,
        parameters.Nc
    )

    delta_feasible, delta_residual = (
        inequality_satisfied(
            A_du,
            b_du,
            U_hold
        )
    )

    Delta_hold = (
        D @ U_hold
        - b_offset
    )

    print(
        "||Delta_U|| =",
        np.linalg.norm(
            Delta_hold
        )
    )

    print(
        "Delta constraints satisfied =",
        delta_feasible
    )

    assert np.allclose(
        Delta_hold,
        0.0
    )

    assert delta_feasible

    print()
    print(
        "Delta-U numerical feasibility: PASSED"
    )

    # ========================================================
    # 10. Delta-U Violation Detection
    # ========================================================

    print()
    print("[10] Delta-U Violation Detection")
    print("-" * 70)

    U_violate = U_hold.copy()

    # Force first thrust increment above du_max
    U_violate[0] = (
        u_prev_test[0]
        + parameters.du_max
        + 0.1
    )

    violation_ok, violation_residual = (
        inequality_satisfied(
            A_du,
            b_du,
            U_violate
        )
    )

    max_violation = np.max(
        violation_residual
    )

    print(
        "Violation detected =",
        not violation_ok
    )

    print(
        "Maximum violation =",
        max_violation
    )

    assert not violation_ok

    assert (
        max_violation > 0.0
    )

    print()
    print(
        "Delta-U violation detection: PASSED"
    )

    # ========================================================
    # 11. State Inequalities
    # ========================================================

    print()
    print("[11] State Inequalities")
    print("-" * 70)

    x0 = np.zeros(
        parameters.nx
    )

    A_x, b_x = (
        constraints.build_state_inequalities(
            Phi,
            Gamma,
            x0
        )
    )

    # 3 finite upper bounds per step
    # 3 finite lower bounds per step
    #
    # 6 * 20 = 120

    expected_state_inequalities = (
        6 * parameters.Np
    )

    expected_A_x_shape = (
        expected_state_inequalities,
        nU
    )

    expected_b_x_shape = (
        expected_state_inequalities,
    )

    print("A_x shape =", A_x.shape)
    print("b_x shape =", b_x.shape)

    assert (
        A_x.shape
        == expected_A_x_shape
    )

    assert (
        b_x.shape
        == expected_b_x_shape
    )

    # Infinite limits must have been removed
    assert np.all(
        np.isfinite(b_x)
    )

    print(
        "All state RHS values finite =",
        np.all(
            np.isfinite(b_x)
        )
    )

    print()
    print("State inequalities validation: PASSED")

    # ========================================================
    # 12. Zero-Control State Feasibility
    # ========================================================

    print()
    print("[12] Zero-Control State Feasibility")
    print("-" * 70)

    U_zero = np.zeros(
        nU
    )

    state_feasible, state_residual = (
        inequality_satisfied(
            A_x,
            b_x,
            U_zero
        )
    )

    X_zero = (
        Phi @ x0
        + Gamma @ U_zero
    )

    print(
        "||X|| =",
        np.linalg.norm(
            X_zero
        )
    )

    print(
        "State constraints satisfied =",
        state_feasible
    )

    assert np.allclose(
        X_zero,
        0.0
    )

    assert state_feasible

    print()
    print(
        "Zero-control state feasibility: PASSED"
    )

    # ========================================================
    # Rotor Feasibility Inequalities
    # ========================================================

    print()
    print("[13] Rotor Feasibility Inequalities")
    print("-" * 70)

    A_rotor, b_rotor = (
        constraints.build_rotor_inequalities()
    )

    expected_rotor_rows = (
        2 * parameters.Nc * parameters.nu
    )

    expected_rotor_shape = (
        expected_rotor_rows,
        nU
    )

    print(
        "A_rotor shape =",
        A_rotor.shape
    )

    print(
        "b_rotor shape =",
        b_rotor.shape
    )

    assert (
        A_rotor.shape
        == expected_rotor_shape
    )

    assert (
        b_rotor.shape
        == (expected_rotor_rows,)
    )

    # ========================================================
    # Soft-State Constraint Structure
    # ========================================================

    print()
    print("[13S] Soft-State Constraint Structure")
    print("-" * 70)

    n_eps = (
        constraints.get_num_slack_variables()
    )

    print(
        "Number of slack variables =",
        n_eps
    )

    assert n_eps == (
        3 * parameters.Np
    )

    A_soft, b_soft = (
        constraints.build_soft_state_inequalities(
            Phi,
            Gamma,
            x0
        )
    )

    A_eps, b_eps = (
        constraints.build_slack_inequalities()
    )

    nZ = (
        nU
        + n_eps
    )

    print(
        "Augmented decision size =",
        nZ
    )

    print(
        "A_soft shape =",
        A_soft.shape
    )

    print(
        "b_soft shape =",
        b_soft.shape
    )

    print(
        "A_eps shape =",
        A_eps.shape
    )

    print(
        "b_eps shape =",
        b_eps.shape
    )

    assert A_soft.shape == (
        6 * parameters.Np,
        nZ
    )

    assert b_soft.shape == (
        6 * parameters.Np,
    )

    assert A_eps.shape == (
        n_eps,
        nZ
    )

    assert b_eps.shape == (
        n_eps,
    )

    # ========================================================
    # Augmented Soft-QP Constraints
    # ========================================================

    print()
    print("[13A] Augmented Soft-QP Constraints")
    print("-" * 70)

    A_soft_qp, b_soft_qp = (
        constraints.build_soft_qp_inequalities(
            Phi,
            Gamma,
            x0,
            np.zeros(
                parameters.nu
            ),
            D
        )
    )

    expected_soft_qp_rows = (
        2 * nU                  # input
        + 2 * nU               # Delta-U
        + 2 * nU               # rotor
        + 6 * parameters.Np     # soft state
        + n_eps                # epsilon >= 0
    )

    expected_soft_qp_cols = (
        nU
        + n_eps
    )

    print(
        "A_soft_qp shape =",
        A_soft_qp.shape
    )

    print(
        "b_soft_qp shape =",
        b_soft_qp.shape
    )

    print(
        "Expected rows =",
        expected_soft_qp_rows
    )

    print(
        "Expected cols =",
        expected_soft_qp_cols
    )

    assert A_soft_qp.shape == (
        expected_soft_qp_rows,
        expected_soft_qp_cols
    )

    assert b_soft_qp.shape == (
        expected_soft_qp_rows,
    )

    # --------------------------------------------------------
    # Zero equilibrium must remain feasible.
    # --------------------------------------------------------

    Z_zero_soft = np.zeros(
        expected_soft_qp_cols,
        dtype=float
    )

    zero_aug_feasible = np.all(
        A_soft_qp @ Z_zero_soft
        <= b_soft_qp + 1.0e-10
    )

    print(
        "Zero augmented QP feasible =",
        zero_aug_feasible
    )

    assert zero_aug_feasible

    print()
    print(
        "Augmented soft-QP structure: PASSED"
    )

    # ========================================================
    # Soft Pitch-Violation Recovery
    # ========================================================

    print()
    print("[13B] Soft Pitch-Violation Recovery")
    print("-" * 70)

    x_pitch_violate = np.zeros(
        parameters.nx,
        dtype=float
    )

    pitch_violation_amount = 0.10

    x_pitch_violate[4] = (
        parameters.theta_max
        + pitch_violation_amount
    )

    A_pitch_soft, b_pitch_soft = (
        constraints.build_soft_qp_inequalities(
            Phi,
            Gamma,
            x_pitch_violate,
            np.zeros(
                parameters.nu
            ),
            D
        )
    )

    Z_pitch_soft = np.zeros(
        nU + n_eps,
        dtype=float
    )

    # Slack ordering per prediction step:
    #
    # [eps_z, eps_phi, eps_theta]
    #
    eps_test = np.zeros(
        (
            parameters.Np,
            3
        ),
        dtype=float
    )

    eps_test[:, 2] = (
        pitch_violation_amount
        + 1.0e-6
    )

    Z_pitch_soft[nU:] = (
        eps_test.reshape(-1)
    )

    pitch_soft_feasible = np.all(
        A_pitch_soft @ Z_pitch_soft
        <= b_pitch_soft + 1.0e-9
    )

    print(
        "Pitch beyond hard limit =",
        x_pitch_violate[4]
    )

    print(
        "Applied theta slack =",
        eps_test[0, 2]
    )

    print(
        "Pitch violation recoverable =",
        pitch_soft_feasible
    )

    assert pitch_soft_feasible

    print()
    print(
        "Soft pitch recovery: PASSED"
    )

    # --------------------------------------------------------
    # Zero state, zero control, zero slack
    # must remain feasible.
    # --------------------------------------------------------

    Z_zero = np.zeros(
        nZ,
        dtype=float
    )

    soft_zero_feasible = np.all(
        A_soft @ Z_zero
        <= b_soft + 1.0e-10
    )

    slack_zero_feasible = np.all(
        A_eps @ Z_zero
        <= b_eps + 1.0e-10
    )

    print(
        "Zero soft-state feasible =",
        soft_zero_feasible
    )

    print(
        "Zero slack feasible =",
        slack_zero_feasible
    )

    assert soft_zero_feasible
    assert slack_zero_feasible

    print()
    print(
        "Soft-state constraint structure: PASSED"
    )

    # --------------------------------------------------------
    # Hover feasibility
    #
    # MPC coordinates:
    #
    #   u = [delta_T, tau_phi, tau_theta, tau_psi]
    #
    # Hover therefore corresponds to:
    #
    #   u = [0, 0, 0, 0]
    # --------------------------------------------------------

    U_hover = np.zeros(
        nU,
        dtype=float
    )

    hover_feasible = np.all(
        A_rotor @ U_hover
        <= b_rotor + 1.0e-9
    )

    print(
        "Hover rotor feasible =",
        hover_feasible
    )

    assert hover_feasible

    # --------------------------------------------------------
    # Roll +0.5 N.m
    #
    # Known feasible at hover for the DD21 geometry.
    # --------------------------------------------------------

    u_roll = np.array(
        [
            0.0,
            0.5,
            0.0,
            0.0
        ],
        dtype=float
    )

    U_roll = np.tile(
        u_roll,
        parameters.Nc
    )

    roll_feasible = np.all(
        A_rotor @ U_roll
        <= b_rotor + 1.0e-9
    )

    print(
        "Roll +0.5 rotor feasible =",
        roll_feasible
    )

    assert roll_feasible

    # --------------------------------------------------------
    # Pitch +0.5 N.m
    #
    # Known infeasible at hover:
    # inverse allocation requires negative w_i^2.
    # --------------------------------------------------------

    u_pitch_bad = np.array(
        [
            0.0,
            0.0,
            0.5,
            0.0
        ],
        dtype=float
    )

    U_pitch_bad = np.tile(
        u_pitch_bad,
        parameters.Nc
    )

    pitch_violation = np.any(
        A_rotor @ U_pitch_bad
        > b_rotor + 1.0e-9
    )

    print(
        "Pitch +0.5 violation detected =",
        pitch_violation
    )

    assert pitch_violation

    # --------------------------------------------------------
    # Yaw +0.5 N.m
    #
    # Strongly infeasible for the current k_m/k_f ratio.
    # --------------------------------------------------------

    u_yaw_bad = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.5
        ],
        dtype=float
    )

    U_yaw_bad = np.tile(
        u_yaw_bad,
        parameters.Nc
    )

    yaw_violation = np.any(
        A_rotor @ U_yaw_bad
        > b_rotor + 1.0e-9
    )

    print(
        "Yaw +0.5 violation detected =",
        yaw_violation
    )

    assert yaw_violation

    print()
    print(
        "Rotor feasibility validation: PASSED"
    )

    # ========================================================
    # 13. Complete QP Inequalities
    # ========================================================

    print()
    print("[13] Complete QP Inequalities")
    print("-" * 70)

    u_prev_zero = np.zeros(
        parameters.nu
    )

    A_ineq, b_ineq = (
        constraints.build_qp_inequalities(
            Phi,
            Gamma,
            x0,
            u_prev_zero,
            D
        )
    )

    # Input:
    #
    # 2 * 80 = 160
    #
    # Delta-U:
    #
    # 2 * 80 = 160
    #
    # Rotor feasibility:
    #
    # 2 * 80 = 160
    #
    # State:
    #
    # 6 * 20 = 120
    #
    # Total:
    #
    # 160 + 160 + 160 + 120 = 600

    expected_input_constraints = (
        2 * nU
    )

    expected_delta_constraints = (
        2 * nU
    )

    expected_rotor_constraints = (
        2 * nU
    )

    expected_state_constraints = (
        6 * parameters.Np
    )

    expected_total_constraints = (
        expected_input_constraints
        + expected_delta_constraints
        + expected_rotor_constraints
        + expected_state_constraints
    )

    expected_A_shape = (
        expected_total_constraints,
        nU
    )

    expected_b_shape = (
        expected_total_constraints,
    )

    print(
        "A_ineq shape =",
        A_ineq.shape
    )

    print(
        "b_ineq shape =",
        b_ineq.shape
    )

    print(
        "Expected constraints =",
        expected_total_constraints
    )

    assert (
        A_ineq.shape
        == expected_A_shape
    )

    assert (
        b_ineq.shape
        == expected_b_shape
    )

    assert np.all(
        np.isfinite(
            b_ineq
        )
    )

    print()
    print(
        "Complete QP dimensions: PASSED"
    )

    # ========================================================
    # 14. Complete QP Feasibility
    # ========================================================

    print()
    print("[14] Complete QP Feasibility")
    print("-" * 70)

    # x0 = 0
    # u_prev = 0
    # U = 0
    #
    # Must satisfy:
    #
    # Input constraints
    # Delta-U constraints
    # Rotor feasibility constraints
    # State constraints

    full_feasible, full_residual = (
        inequality_satisfied(
            A_ineq,
            b_ineq,
            U_zero
        )
    )

    max_full_residual = np.max(
        full_residual
    )

    print(
        "Zero U satisfies all constraints =",
        full_feasible
    )

    print(
        "Maximum inequality residual =",
        max_full_residual
    )

    assert full_feasible

    print()
    print(
        "Complete QP feasibility: PASSED"
    )

    # ========================================================
    # 15. Input Violation Detection
    # ========================================================

    print()
    print("[15] Input Violation Detection")
    print("-" * 70)

    U_input_violate = np.zeros(
        nU
    )

    U_input_violate[0] = (
        parameters.thrust_max
        + 1.0
    )

    input_ok, input_residual = (
        inequality_satisfied(
            A_u,
            b_u,
            U_input_violate
        )
    )

    print(
        "Input violation detected =",
        not input_ok
    )

    print(
        "Maximum input violation =",
        np.max(
            input_residual
        )
    )

    assert not input_ok

    print()
    print(
        "Input violation detection: PASSED"
    )

    # ========================================================
    # 16. Direct Stacking Verification
    # ========================================================

    print()
    print("[16] Direct Stacking Verification")
    print("-" * 70)

    A_du_zero, b_du_zero = (
        constraints.build_delta_inequalities(
            D,
            u_prev_zero
        )
    )

    A_expected = np.vstack(
        (
            A_u,
            A_du_zero,
            A_rotor,
            A_x
        )
    )

    b_expected = np.concatenate(
        (
            b_u,
            b_du_zero,
            b_rotor,
            b_x
        )
    )
    A_correct = np.allclose(
        A_ineq,
        A_expected
    )

    b_correct = np.allclose(
        b_ineq,
        b_expected
    )

    print(
        "A_ineq stacking correct =",
        A_correct
    )

    print(
        "b_ineq stacking correct =",
        b_correct
    )

    assert A_correct
    assert b_correct

    print()
    print(
        "Direct stacking verification: PASSED"
    )

    # ========================================================
    # 17. Final Results
    # ========================================================

    print()
    print("=" * 70)
    print(" CONSTRAINTS TEST RESULTS")
    print("=" * 70)

    print(
        "Phi    :",
        Phi.shape
    )

    print(
        "Gamma  :",
        Gamma.shape
    )

    print(
        "D      :",
        D.shape
    )

    print(
        "U_min  :",
        U_min.shape
    )

    print(
        "U_max  :",
        U_max.shape
    )

    print(
        "X_min  :",
        X_min.shape
    )

    print(
        "X_max  :",
        X_max.shape
    )

    print(
        "A_u    :",
        A_u.shape
    )

    print(
        "A_du   :",
        A_du.shape
    )

    print(
        "A_rotor:",
        A_rotor.shape
    )

    print(
        "A_x    :",
        A_x.shape
    )

    print(
        "A_ineq :",
        A_ineq.shape
    )

    print(
        "b_ineq :",
        b_ineq.shape
    )

    print()
    print(
        "Expected total inequalities =",
        expected_total_constraints
    )

    print()
    print(
        "ALL CONSTRAINT TESTS PASSED"
    )

    print("=" * 70)
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()