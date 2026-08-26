#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_cost_function.py

Validation test for the quadratic cost function
of the Physics-based MPC controller
for the Duckiedrone DD21.

The test validates:

1. MPC parameters
2. Prediction matrices Phi and Gamma
3. Augmented state weighting matrix Qbar
4. Augmented input weighting matrix Rbar
5. Augmented control increment weighting matrix Sbar
6. Control increment matrix D
7. Control increment offset vector b
8. Delta_U = D @ U - b
9. Reference trajectory construction
10. QP Hessian H
11. QP gradient f
12. Hessian symmetry
13. Hessian positive definiteness
14. Zero-reference / zero-u_prev gradient
15. Nonzero-u_prev gradient contribution
16. Direct analytical verification

QP formulation:

    J(U) = 0.5 * U.T @ H @ U + f.T @ U

Prediction model:

    X = Phi @ x0 + Gamma @ U

Control increments:

    Delta_U = D @ U - b

Cost:

    J =
        (X - X_ref).T @ Qbar @ (X - X_ref)
        + U.T @ Rbar @ U
        + Delta_U.T @ Sbar @ Delta_U

QP matrices:

    H = 2 * (
        Gamma.T @ Qbar @ Gamma
        + Rbar
        + D.T @ Sbar @ D
    )

    f = (
        2 * Gamma.T @ Qbar @
        (Phi @ x0 - X_ref)
        - 2 * D.T @ Sbar @ b
    )

Author: Abdallah GHOUL 2026
"""

import numpy as np

from .parameters import MPCParameters
from .physics_model import PhysicsModel
from .cost_function import CostFunction


# ============================================================
# Main Test
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" Duckiedrone DD21 - Physics MPC")
    print(" Cost Function Validation")
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
    print("Q shape =", parameters.Q.shape)
    print("R shape =", parameters.R.shape)
    print("S shape =", parameters.S.shape)
    print("P shape =", parameters.P.shape)

    assert parameters.nx == 12, (
        f"Expected nx = 12, got {parameters.nx}"
    )

    assert parameters.nu == 4, (
        f"Expected nu = 4, got {parameters.nu}"
    )

    assert parameters.Np == 20, (
        f"Expected Np = 20, got {parameters.Np}"
    )

    assert parameters.Nc == 20, (
        f"Expected Nc = 20, got {parameters.Nc}"
    )

    assert parameters.Q.shape == (
        parameters.nx,
        parameters.nx
    )

    assert parameters.R.shape == (
        parameters.nu,
        parameters.nu
    )

    assert parameters.S.shape == (
        parameters.nu,
        parameters.nu
    )

    assert parameters.P.shape == (
        parameters.nx,
        parameters.nx
    )

    print()
    print("Parameters validation: PASSED")

    # ========================================================
    # 2. Physics Prediction Model
    # ========================================================

    print()
    print("[2] Physics Prediction Model")
    print("-" * 70)

    model = PhysicsModel(parameters)

    Phi = model.build_phi()
    Gamma = model.build_gamma()

    print("Phi shape   =", Phi.shape)
    print("Gamma shape =", Gamma.shape)

    expected_phi_shape = (
        parameters.Np * parameters.nx,
        parameters.nx
    )

    expected_gamma_shape = (
        parameters.Np * parameters.nx,
        parameters.Nc * parameters.nu
    )

    assert Phi.shape == expected_phi_shape, (
        f"Invalid Phi shape: "
        f"expected {expected_phi_shape}, "
        f"got {Phi.shape}"
    )

    assert Gamma.shape == expected_gamma_shape, (
        f"Invalid Gamma shape: "
        f"expected {expected_gamma_shape}, "
        f"got {Gamma.shape}"
    )

    print()
    print("Prediction matrices validation: PASSED")

    # ========================================================
    # 3. Cost Function Object
    # ========================================================

    print()
    print("[3] Cost Function")
    print("-" * 70)

    cost = CostFunction(parameters)

    print("CostFunction created successfully.")

    # ========================================================
    # 4. Qbar Validation
    # ========================================================

    print()
    print("[4] Qbar Validation")
    print("-" * 70)

    Qbar = cost.build_qbar()

    print("Qbar :", Qbar.shape)

    expected_qbar_shape = (
        parameters.Np * parameters.nx,
        parameters.Np * parameters.nx
    )

    assert Qbar.shape == expected_qbar_shape, (
        f"Invalid Qbar shape: "
        f"expected {expected_qbar_shape}, "
        f"got {Qbar.shape}"
    )

    qbar_symmetric = np.allclose(
        Qbar,
        Qbar.T
    )

    print("Qbar symmetric =", qbar_symmetric)

    assert qbar_symmetric, (
        "Qbar is not symmetric."
    )

    # First Q block
    Q_first = Qbar[
        0:parameters.nx,
        0:parameters.nx
    ]

    first_q_correct = np.allclose(
        Q_first,
        parameters.Q
    )

    print(
        "First Q block correct =",
        first_q_correct
    )

    assert first_q_correct, (
        "First diagonal block of Qbar is not Q."
    )

    # Terminal P block
    terminal_start = (
        (parameters.Np - 1)
        * parameters.nx
    )

    P_terminal = Qbar[
        terminal_start:
        terminal_start + parameters.nx,

        terminal_start:
        terminal_start + parameters.nx
    ]

    terminal_correct = np.allclose(
        P_terminal,
        parameters.P
    )

    print(
        "Terminal P block correct =",
        terminal_correct
    )

    assert terminal_correct, (
        "Terminal block of Qbar is not P."
    )

    print()
    print("Qbar validation: PASSED")

    # ========================================================
    # 5. Rbar Validation
    # ========================================================

    print()
    print("[5] Rbar Validation")
    print("-" * 70)

    Rbar = cost.build_rbar()

    print("Rbar :", Rbar.shape)

    expected_rbar_shape = (
        parameters.Nc * parameters.nu,
        parameters.Nc * parameters.nu
    )

    assert Rbar.shape == expected_rbar_shape, (
        f"Invalid Rbar shape: "
        f"expected {expected_rbar_shape}, "
        f"got {Rbar.shape}"
    )

    rbar_symmetric = np.allclose(
        Rbar,
        Rbar.T
    )

    print("Rbar symmetric =", rbar_symmetric)

    assert rbar_symmetric, (
        "Rbar is not symmetric."
    )

    # First R block
    R_first = Rbar[
        0:parameters.nu,
        0:parameters.nu
    ]

    first_r_correct = np.allclose(
        R_first,
        parameters.R
    )

    print(
        "First R block correct =",
        first_r_correct
    )

    assert first_r_correct, (
        "First diagonal block of Rbar is not R."
    )

    # Last R block
    last_r_start = (
        (parameters.Nc - 1)
        * parameters.nu
    )

    R_last = Rbar[
        last_r_start:
        last_r_start + parameters.nu,

        last_r_start:
        last_r_start + parameters.nu
    ]

    last_r_correct = np.allclose(
        R_last,
        parameters.R
    )

    print(
        "Last R block correct =",
        last_r_correct
    )

    assert last_r_correct, (
        "Last diagonal block of Rbar is not R."
    )

    print()
    print("Rbar validation: PASSED")

    # ========================================================
    # 6. Sbar Validation
    # ========================================================

    print()
    print("[6] Sbar Validation")
    print("-" * 70)

    Sbar = cost.build_sbar()

    print("Sbar :", Sbar.shape)

    expected_sbar_shape = (
        parameters.Nc * parameters.nu,
        parameters.Nc * parameters.nu
    )

    assert Sbar.shape == expected_sbar_shape, (
        f"Invalid Sbar shape: "
        f"expected {expected_sbar_shape}, "
        f"got {Sbar.shape}"
    )

    sbar_symmetric = np.allclose(
        Sbar,
        Sbar.T
    )

    print("Sbar symmetric =", sbar_symmetric)

    assert sbar_symmetric, (
        "Sbar is not symmetric."
    )

    # First S block
    S_first = Sbar[
        0:parameters.nu,
        0:parameters.nu
    ]

    first_s_correct = np.allclose(
        S_first,
        parameters.S
    )

    print(
        "First S block correct =",
        first_s_correct
    )

    assert first_s_correct, (
        "First diagonal block of Sbar is not S."
    )

    # Last S block
    last_s_start = (
        (parameters.Nc - 1)
        * parameters.nu
    )

    S_last = Sbar[
        last_s_start:
        last_s_start + parameters.nu,

        last_s_start:
        last_s_start + parameters.nu
    ]

    last_s_correct = np.allclose(
        S_last,
        parameters.S
    )

    print(
        "Last S block correct =",
        last_s_correct
    )

    assert last_s_correct, (
        "Last diagonal block of Sbar is not S."
    )

    print()
    print("Sbar validation: PASSED")

    # ========================================================
    # 7. Delta Matrix D Validation
    # ========================================================

    print()
    print("[7] Delta Matrix D Validation")
    print("-" * 70)

    D = cost.build_delta_matrix()

    expected_d_shape = (
        parameters.Nc * parameters.nu,
        parameters.Nc * parameters.nu
    )

    print("D shape =", D.shape)

    assert D.shape == expected_d_shape, (
        f"Invalid D shape: "
        f"expected {expected_d_shape}, "
        f"got {D.shape}"
    )

    Iu = np.eye(parameters.nu)

    # First diagonal block must be I
    D00 = D[
        0:parameters.nu,
        0:parameters.nu
    ]

    first_d_correct = np.allclose(
        D00,
        Iu
    )

    print(
        "D first diagonal block = I :",
        first_d_correct
    )

    assert first_d_correct, (
        "First diagonal block of D is not I."
    )

    # Second diagonal block must be I
    D11 = D[
        parameters.nu:
        2 * parameters.nu,

        parameters.nu:
        2 * parameters.nu
    ]

    second_diagonal_correct = np.allclose(
        D11,
        Iu
    )

    print(
        "D second diagonal block = I :",
        second_diagonal_correct
    )

    assert second_diagonal_correct, (
        "Second diagonal block of D is not I."
    )

    # First lower diagonal block must be -I
    D10 = D[
        parameters.nu:
        2 * parameters.nu,

        0:
        parameters.nu
    ]

    lower_block_correct = np.allclose(
        D10,
        -Iu
    )

    print(
        "D first lower block = -I :",
        lower_block_correct
    )

    assert lower_block_correct, (
        "First lower diagonal block of D is not -I."
    )

    print()
    print("Delta matrix D validation: PASSED")

    # ========================================================
    # 8. Delta Offset b Validation
    # ========================================================

    print()
    print("[8] Delta Offset b Validation")
    print("-" * 70)

    u_prev_test = np.array([
        0.50,
        -0.20,
        0.10,
        0.05
    ])

    b = cost.build_delta_offset(
        u_prev_test
    )

    expected_b_shape = (
        parameters.Nc * parameters.nu,
    )

    print("b shape =", b.shape)

    assert b.shape == expected_b_shape, (
        f"Invalid b shape: "
        f"expected {expected_b_shape}, "
        f"got {b.shape}"
    )

    first_b_correct = np.allclose(
        b[:parameters.nu],
        u_prev_test
    )

    remaining_b_zero = np.allclose(
        b[parameters.nu:],
        0.0
    )

    print(
        "b first block = u_prev :",
        first_b_correct
    )

    print(
        "b remaining blocks zero :",
        remaining_b_zero
    )

    assert first_b_correct, (
        "First block of b is not u_prev."
    )

    assert remaining_b_zero, (
        "Remaining blocks of b are not zero."
    )

    print()
    print("Delta offset b validation: PASSED")

    # ========================================================
    # 9. Delta_U Numerical Validation
    # ========================================================

    print()
    print("[9] Delta_U Numerical Validation")
    print("-" * 70)

    # Build an artificial control sequence
    U_steps = np.zeros(
        (
            parameters.Nc,
            parameters.nu
        ),
        dtype=float
    )

    U_steps[0] = np.array([
        0.60,
        -0.10,
        0.10,
        0.10
    ])

    U_steps[1] = np.array([
        0.70,
        -0.15,
        0.20,
        0.10
    ])

    # Keep remaining controls equal to u1
    for k in range(2, parameters.Nc):
        U_steps[k] = U_steps[1]

    U_test = U_steps.reshape(-1)

    Delta_U = (
        D @ U_test
        - b
    )

    # Build expected Delta_U manually
    Delta_expected_steps = np.zeros_like(
        U_steps
    )

    Delta_expected_steps[0] = (
        U_steps[0]
        - u_prev_test
    )

    for k in range(
        1,
        parameters.Nc
    ):
        Delta_expected_steps[k] = (
            U_steps[k]
            - U_steps[k - 1]
        )

    Delta_expected = (
        Delta_expected_steps.reshape(-1)
    )

    delta_correct = np.allclose(
        Delta_U,
        Delta_expected
    )

    print(
        "Delta_U = D @ U - b correct =",
        delta_correct
    )

    print()
    print(
        "Delta u0 =",
        Delta_U[
            0:parameters.nu
        ]
    )

    print(
        "Expected  =",
        U_steps[0] - u_prev_test
    )

    print()
    print(
        "Delta u1 =",
        Delta_U[
            parameters.nu:
            2 * parameters.nu
        ]
    )

    print(
        "Expected  =",
        U_steps[1] - U_steps[0]
    )

    assert delta_correct, (
        "Delta_U does not match "
        "the expected control increments."
    )

    print()
    print("Delta_U numerical validation: PASSED")

    # ========================================================
    # 10. Reference Vector Validation
    # ========================================================

    print()
    print("[10] Reference Vector Validation")
    print("-" * 70)

    x_ref = np.zeros(
        parameters.nx
    )

    X_ref = cost.build_reference(
        x_ref
    )

    expected_reference_shape = (
        parameters.Np
        * parameters.nx,
    )

    print("x_ref shape =", x_ref.shape)
    print("X_ref shape =", X_ref.shape)

    assert X_ref.shape == expected_reference_shape, (
        f"Invalid X_ref shape: "
        f"expected {expected_reference_shape}, "
        f"got {X_ref.shape}"
    )

    assert np.allclose(
        X_ref,
        0.0
    ), (
        "Zero reference was not "
        "stacked correctly."
    )

    print()
    print("Reference validation: PASSED")

    # ========================================================
    # 11. QP Matrix Validation
    # ========================================================

    print()
    print("[11] QP Matrix Validation")
    print("-" * 70)

    x0 = np.zeros(
        parameters.nx
    )

    u_prev = np.zeros(
        parameters.nu
    )

    H, f = cost.build_qp_matrices(
        Phi,
        Gamma,
        x0,
        x_ref,
        u_prev
    )

    print("H shape =", H.shape)
    print("f shape =", f.shape)

    expected_h_shape = (
        parameters.Nc * parameters.nu,
        parameters.Nc * parameters.nu
    )

    expected_f_shape = (
        parameters.Nc * parameters.nu,
    )

    assert H.shape == expected_h_shape, (
        f"Invalid H shape: "
        f"expected {expected_h_shape}, "
        f"got {H.shape}"
    )

    assert f.shape == expected_f_shape, (
        f"Invalid f shape: "
        f"expected {expected_f_shape}, "
        f"got {f.shape}"
    )

    print()
    print("QP dimensions validation: PASSED")

    # ========================================================
    # 12. Hessian Symmetry
    # ========================================================

    print()
    print("[12] Hessian Symmetry")
    print("-" * 70)

    H_symmetric = np.allclose(
        H,
        H.T
    )

    print(
        "H symmetric =",
        H_symmetric
    )

    assert H_symmetric, (
        "Hessian H is not symmetric."
    )

    print()
    print("Hessian symmetry validation: PASSED")

    # ========================================================
    # 13. Hessian Eigenvalues
    # ========================================================

    print()
    print("[13] Hessian Eigenvalues")
    print("-" * 70)

    eigvals = np.linalg.eigvalsh(
        H
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

    positive_definite = (
        min_eig > 0.0
    )

    print(
        "H positive definite =",
        positive_definite
    )

    assert positive_definite, (
        "Hessian H is not positive definite."
    )

    print()
    print(
        "Hessian positive definiteness: PASSED"
    )

    # ========================================================
    # 14. Zero Gradient Validation
    # ========================================================

    print()
    print("[14] Zero Gradient Validation")
    print("-" * 70)

    # x0 = 0
    # x_ref = 0
    # u_prev = 0
    #
    # therefore:
    #
    # e0 = 0
    # b  = 0
    # f  = 0

    f_norm = np.linalg.norm(
        f
    )

    gradient_zero = np.allclose(
        f,
        np.zeros_like(f),
        atol=1e-12
    )

    print("||f|| =", f_norm)

    print(
        "f approximately zero =",
        gradient_zero
    )

    assert gradient_zero, (
        "Gradient f should be zero for "
        "x0 = 0, x_ref = 0, u_prev = 0."
    )

    print()
    print("Zero gradient validation: PASSED")

    # ========================================================
    # 15. Nonzero u_prev Gradient Validation
    # ========================================================

    print()
    print("[15] Nonzero u_prev Gradient Validation")
    print("-" * 70)

    u_prev_nonzero = np.array([
        0.50,
        -0.20,
        0.10,
        0.05
    ])

    H_nonzero, f_nonzero = (
        cost.build_qp_matrices(
            Phi,
            Gamma,
            x0,
            x_ref,
            u_prev_nonzero
        )
    )

    b_nonzero = cost.build_delta_offset(
        u_prev_nonzero
    )

    # Since x0 = 0 and x_ref = 0:
    #
    # f = -2 D.T Sbar b

    f_nonzero_expected = (
        -2.0
        * (
            D.T
            @ Sbar
            @ b_nonzero
        )
    )

    nonzero_gradient_correct = np.allclose(
        f_nonzero,
        f_nonzero_expected
    )

    print(
        "Nonzero u_prev gradient correct =",
        nonzero_gradient_correct
    )

    print(
        "||f_nonzero|| =",
        np.linalg.norm(f_nonzero)
    )

    assert nonzero_gradient_correct, (
        "Gradient contribution from "
        "u_prev is incorrect."
    )

    # H must not depend on u_prev
    h_independent_of_u_prev = np.allclose(
        H,
        H_nonzero
    )

    print(
        "H independent of u_prev =",
        h_independent_of_u_prev
    )

    assert h_independent_of_u_prev, (
        "H should not depend on u_prev."
    )

    print()
    print(
        "Nonzero u_prev gradient validation: PASSED"
    )

    # ========================================================
    # 16. Direct Mathematical Verification
    # ========================================================

    print()
    print("[16] Direct Mathematical Verification")
    print("-" * 70)

    # --------------------------------------------------------
    # H analytical verification
    # --------------------------------------------------------

    H_expected = 2.0 * (
        Gamma.T
        @ Qbar
        @ Gamma
        + Rbar
        + D.T
        @ Sbar
        @ D
    )

    H_expected = 0.5 * (
        H_expected
        + H_expected.T
    )

    H_correct = np.allclose(
        H,
        H_expected
    )

    print(
        "H matches analytical formula =",
        H_correct
    )

    assert H_correct, (
        "H does not match "
        "the analytical formula."
    )

    # --------------------------------------------------------
    # f analytical verification
    # Zero u_prev case
    # --------------------------------------------------------

    e0 = (
        Phi @ x0
        - X_ref
    )

    b_zero = cost.build_delta_offset(
        u_prev
    )

    f_expected = (
        2.0
        * (
            Gamma.T
            @ Qbar
            @ e0
        )
        -
        2.0
        * (
            D.T
            @ Sbar
            @ b_zero
        )
    )

    f_correct = np.allclose(
        f,
        f_expected
    )

    print(
        "f matches analytical formula =",
        f_correct
    )

    assert f_correct, (
        "f does not match "
        "the analytical formula."
    )

    print()
    print(
        "Analytical QP verification: PASSED"
    )

    # ========================================================
    # 17. Soft-QP Hessian / Gradient Validation
    # ========================================================

    print()
    print("[17] Soft-QP Hessian / Gradient Validation")
    print("-" * 70)

    # --------------------------------------------------------
    # Augmented decision-vector dimensions
    #
    # Z = [U; epsilon]
    #
    # U       : Nc * nu = 80
    # epsilon : 3 * Np  = 60
    # Z       : 140
    # --------------------------------------------------------

    n_u = (
        parameters.Nc
        * parameters.nu
    )

    n_slack = (
        3
        * parameters.Np
    )

    n_z = (
        n_u
        + n_slack
    )

    print("n_u     =", n_u)
    print("n_slack =", n_slack)
    print("n_z     =", n_z)

    assert n_u == 80, (
        f"Expected n_u = 80, got {n_u}"
    )

    assert n_slack == 60, (
        f"Expected n_slack = 60, got {n_slack}"
    )

    assert n_z == 140, (
        f"Expected n_z = 140, got {n_z}"
    )

    # --------------------------------------------------------
    # Build augmented Soft-QP matrices
    # --------------------------------------------------------

    H_soft, f_soft = (
        cost.build_soft_qp_matrices(
            Phi,
            Gamma,
            x0,
            x_ref,
            u_prev,
            n_slack
        )
    )

    print()
    print("H_soft shape =", H_soft.shape)
    print("f_soft shape =", f_soft.shape)

    expected_h_soft_shape = (
        n_z,
        n_z
    )

    expected_f_soft_shape = (
        n_z,
    )

    assert H_soft.shape == expected_h_soft_shape, (
        f"Invalid H_soft shape: "
        f"expected {expected_h_soft_shape}, "
        f"got {H_soft.shape}"
    )

    assert f_soft.shape == expected_f_soft_shape, (
        f"Invalid f_soft shape: "
        f"expected {expected_f_soft_shape}, "
        f"got {f_soft.shape}"
    )

    # --------------------------------------------------------
    # Hessian symmetry
    # --------------------------------------------------------

    h_soft_symmetric = np.allclose(
        H_soft,
        H_soft.T
    )

    print(
        "H_soft symmetric =",
        h_soft_symmetric
    )

    assert h_soft_symmetric, (
        "Soft-QP Hessian is not symmetric."
    )

    # --------------------------------------------------------
    # Original hard-QP Hessian block
    # --------------------------------------------------------

    H_U_block = H_soft[
        :n_u,
        :n_u
    ]

    original_h_preserved = np.allclose(
        H_U_block,
        H
    )

    print(
        "Original H block preserved =",
        original_h_preserved
    )

    assert original_h_preserved, (
        "Top-left block of H_soft "
        "does not match original H."
    )

    # --------------------------------------------------------
    # Cross blocks must be zero
    # --------------------------------------------------------

    H_U_eps = H_soft[
        :n_u,
        n_u:
    ]

    H_eps_U = H_soft[
        n_u:,
        :n_u
    ]

    upper_cross_zero = np.allclose(
        H_U_eps,
        0.0
    )

    lower_cross_zero = np.allclose(
        H_eps_U,
        0.0
    )

    print(
        "Upper cross block zero =",
        upper_cross_zero
    )

    print(
        "Lower cross block zero =",
        lower_cross_zero
    )

    assert upper_cross_zero, (
        "Upper U-epsilon Hessian block "
        "must be zero."
    )

    assert lower_cross_zero, (
        "Lower epsilon-U Hessian block "
        "must be zero."
    )

    # --------------------------------------------------------
    # Slack Hessian block
    # --------------------------------------------------------

    rho_eps = float(
        parameters.slack_weight
    )

    H_eps = H_soft[
        n_u:,
        n_u:
    ]

    H_eps_expected = (
        2.0
        * rho_eps
        * np.eye(n_slack)
    )

    slack_hessian_correct = np.allclose(
        H_eps,
        H_eps_expected
    )

    print(
        "slack_weight =",
        rho_eps
    )

    print(
        "Slack Hessian block correct =",
        slack_hessian_correct
    )

    assert slack_hessian_correct, (
        "Slack Hessian block must equal "
        "2 * slack_weight * I."
    )

    # --------------------------------------------------------
    # Gradient validation
    # --------------------------------------------------------

    f_U_soft = f_soft[
        :n_u
    ]

    f_eps = f_soft[
        n_u:
    ]

    original_f_preserved = np.allclose(
        f_U_soft,
        f
    )

    slack_gradient_zero = np.allclose(
        f_eps,
        0.0
    )

    print(
        "Original f block preserved =",
        original_f_preserved
    )

    print(
        "Slack gradient zero =",
        slack_gradient_zero
    )

    assert original_f_preserved, (
        "First block of f_soft "
        "does not match original f."
    )

    assert slack_gradient_zero, (
        "Slack-variable gradient "
        "must be zero."
    )

    print()
    print(
        "Soft-QP Hessian/gradient validation: PASSED"
    )

    # ========================================================
    # 18. Final Results
    # ========================================================

    print()
    print("=" * 70)
    print(" COST FUNCTION TEST RESULTS")
    print("=" * 70)

    print(
        "Phi   :",
        Phi.shape
    )

    print(
        "Gamma :",
        Gamma.shape
    )

    print(
        "Qbar  :",
        Qbar.shape
    )

    print(
        "Rbar  :",
        Rbar.shape
    )

    print(
        "Sbar  :",
        Sbar.shape
    )

    print(
        "D     :",
        D.shape
    )

    print(
        "b     :",
        b.shape
    )

    print(
        "H shape =",
        H.shape
    )

    print(
        "f shape =",
        f.shape
    )

    print(
        "H symmetric =",
        H_symmetric
    )

    print(
        "Minimum eigenvalue =",
        min_eig
    )

    print(
        "Maximum eigenvalue =",
        max_eig
    )

    print(
        "||f|| zero case =",
        f_norm
    )

    print(
        "||f|| nonzero u_prev =",
        np.linalg.norm(
            f_nonzero
        )
    )

    print()
    print(
        "ALL COST FUNCTION TESTS PASSED"
    )

    print("=" * 70)
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()