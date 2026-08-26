#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_physics_mpc_controller.py

Standalone numerical validation of the
PhysicsMPCController integration.

This test verifies:

1. Controller initialization
2. Hover-equilibrium conversion
3. Altitude reference response
4. MPC-space previous control storage
5. Physical thrust conversion
6. Controller reset
7. Infeasible-problem fallback

Author: Abdallah GHOUL 2026
"""

import numpy as np
import os, sys
import rospkg
_PKG = rospkg.RosPack().get_path("duckiedrone_validation")
for _d in ("controllers", "models", "scenarios"):
    _p = os.path.join(_PKG, "scripts", _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from physics_mpc_controller import PhysicsMPCController


# ============================================================
# Utility
# ============================================================

def print_section(number, title):

    print()
    print(
        f"[{number}] {title}"
    )

    print("-" * 70)


# ============================================================
# Main Test
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" Duckiedrone DD21 - Physics MPC")
    print(" Controller Integration Validation")
    print("=" * 70)

    # ========================================================
    # 1. Controller Initialization
    # ========================================================

    print_section(
        1,
        "Controller Initialization"
    )

    controller = (
        PhysicsMPCController()
    )

    print(
        "Hover thrust =",
        controller.hover_thrust
    )

    print(
        "Initial u_prev =",
        controller.u_prev_mpc
    )

    assert np.isclose(
        controller.hover_thrust,
        0.635 * 9.81
    )

    assert np.allclose(
        controller.u_prev_mpc,
        np.zeros(4)
    )

    # --------------------------------------------------------
    # ControllerBase previous control must remain
    # in physical wrench coordinates:
    #
    # [T, tau_phi, tau_theta, tau_psi]
    # --------------------------------------------------------

    print(
        "Base physical u_prev =",
        controller.u_prev
    )

    expected_base_hover = np.array(
        [
            controller.hover_thrust,
            0.0,
            0.0,
            0.0
        ],
        dtype=float
    )

    base_previous_control_correct = (
        np.allclose(
            controller.u_prev,
            expected_base_hover,
            atol=1e-6
        )
    )

    print(
        "Base previous control correct =",
        base_previous_control_correct
    )

    assert base_previous_control_correct

    print()
    print(
        "Controller initialization: PASSED"
    )

    # ========================================================
    # 2. Hover Equilibrium Test
    # ========================================================

    print_section(
        2,
        "Hover Equilibrium"
    )

    x = np.zeros(
        12,
        dtype=float
    )

    x_ref = np.zeros(
        12,
        dtype=float
    )

    u_hover = (
        controller.control_law(
            x,
            x_ref
        )
    )

    print(
        "Physical control =",
        u_hover
    )

    print(
        "MPC u_prev =",
        controller.u_prev_mpc
    )

    expected_hover = np.array(
        [
            controller.hover_thrust,
            0.0,
            0.0,
            0.0
        ]
    )

    hover_correct = np.allclose(
        u_hover,
        expected_hover,
        atol=1e-6
    )

    print(
        "Hover physical control correct =",
        hover_correct
    )

    assert hover_correct

    assert np.allclose(
        controller.u_prev_mpc,
        np.zeros(4),
        atol=1e-6
    )

    print()
    print(
        "Hover-equilibrium test: PASSED"
    )

    # ========================================================
    # 3. Altitude Reference
    # ========================================================

    print_section(
        3,
        "Altitude Reference"
    )

    controller.reset()

    x = np.zeros(
        12,
        dtype=float
    )

    x_ref = np.zeros(
        12,
        dtype=float
    )

    x_ref[2] = 1.0

    u_altitude = (
        controller.control_law(
            x,
            x_ref
        )
    )

    print(
        "Physical control =",
        u_altitude
    )

    print(
        "Stored MPC u_prev =",
        controller.u_prev_mpc
    )

    delta_T = (
        controller.u_prev_mpc[0]
    )

    physical_T = (
        u_altitude[0]
    )

    print(
        "delta_T =",
        delta_T
    )

    print(
        "Physical thrust =",
        physical_T
    )

    print(
        "Expected physical thrust =",
        controller.hover_thrust
        + delta_T
    )

    assert delta_T > 0.0

    assert physical_T > (
        controller.hover_thrust
    )

    assert np.isclose(
        physical_T,
        controller.hover_thrust
        + delta_T,
        atol=1e-6
    )

    # --------------------------------------------------------
    # Simulate ControllerBase.spin()
    #
    # ControllerBase performs:
    #
    #   u = control_law(...)
    #   u = clamp_u(u)
    #   self.u_prev = u
    #
    # self.u_prev must therefore remain PHYSICAL,
    # while self.u_prev_mpc remains in deviation coordinates.
    # --------------------------------------------------------

    u_published = (
        controller.clamp_u(
            u_altitude.copy()
        )
    )

    controller.u_prev = (
        u_published.copy()
    )

    print(
        "Published physical control =",
        u_published
    )

    print(
        "Base physical u_prev =",
        controller.u_prev
    )

    print(
        "Internal MPC u_prev =",
        controller.u_prev_mpc
    )

    # Base memory is physical
    assert np.allclose(
        controller.u_prev,
        u_published,
        atol=1e-6
    )

    # --------------------------------------------------------
    # MPC memory must contain the ACTUALLY APPLIED input
    # in deviation coordinates, not merely the optimizer
    # request.
    # --------------------------------------------------------

    applied_mpc = (
        u_published.copy()
    )

    applied_mpc[0] -= (
        controller.hover_thrust
    )

    print(
        "Applied MPC-coordinate control =",
        applied_mpc
    )

    assert np.allclose(
        controller.u_prev_mpc,
        applied_mpc,
        atol=1e-6
    )

    # --------------------------------------------------------
    # Verify that the physical and MPC memories represent
    # the same applied command in different coordinates.
    # --------------------------------------------------------

    assert np.isclose(
        controller.u_prev[0],
        controller.hover_thrust
        + controller.u_prev_mpc[0],
        atol=1e-6
    )

    assert np.allclose(
        controller.u_prev[1:4],
        controller.u_prev_mpc[1:4],
        atol=1e-6
    )

    print(
        "Base/MPC coordinate separation correct = True"
    )

    print()
    print(
        "Altitude-reference test: PASSED"
    )

    # ========================================================
    # 4. Previous-Control Coordinate Validation
    # ========================================================

    print_section(
        4,
        "Previous-Control Coordinate Validation"
    )

    print(
        "u_prev =",
        controller.u_prev_mpc
    )

    print(
        "Physical control =",
        controller.last_physical_control
    )

    reconstructed_physical = (
        controller.u_prev_mpc.copy()
    )

    reconstructed_physical[0] += (
        controller.hover_thrust
    )

    coordinate_conversion_correct = (
        np.allclose(
            reconstructed_physical,
            controller.last_physical_control,
            atol=1e-6
        )
    )

    print(
        "Coordinate conversion correct =",
        coordinate_conversion_correct
    )

    assert coordinate_conversion_correct

    print()
    print(
        "Previous-control validation: PASSED"
    )

    # ========================================================
    # 5. Second MPC Step
    # ========================================================

    print_section(
        5,
        "Second MPC Step"
    )

    previous_u = (
        controller.u_prev_mpc.copy()
    )

    # --------------------------------------------------------
    # Compute requested physical control from the MPC.
    # --------------------------------------------------------

    u_second_requested = (
        controller.control_law(
            x,
            x_ref
        )
    )

    # --------------------------------------------------------
    # Reproduce ControllerBase.spin():
    #
    #   u = control_law(...)
    #   u = clamp_u(u)
    #   self.u_prev = u
    #
    # The clamp synchronizes u_prev_mpc with the actually
    # applied command.
    # --------------------------------------------------------

    u_second_applied = (
        controller.clamp_u(
            u_second_requested.copy()
        )
    )

    controller.u_prev = (
        u_second_applied.copy()
    )

    current_u = (
        controller.u_prev_mpc.copy()
    )

    delta_u = (
        current_u
        - previous_u
    )

    print(
        "Previous MPC input =",
        previous_u
    )

    print(
        "Requested physical control =",
        u_second_requested
    )

    print(
        "Applied physical control =",
        u_second_applied
    )

    print(
        "Current applied MPC input =",
        current_u
    )

    print(
        "Delta_u =",
        delta_u
    )

    # --------------------------------------------------------
    # The stored MPC input must reconstruct the physical
    # command that was actually applied.
    # --------------------------------------------------------

    reconstructed_second = (
        current_u.copy()
    )

    reconstructed_second[0] += (
        controller.hover_thrust
    )

    assert np.allclose(
        reconstructed_second,
        u_second_applied,
        atol=1e-6
    )

    # --------------------------------------------------------
    # Applied rate increment must respect the common physical
    # rate limit used by ControllerBase.
    # --------------------------------------------------------

    assert np.all(
        np.abs(delta_u)
        <= controller.du_max + 1e-6
    )

    print(
        "Applied-command synchronization correct = True"
    )

    print(
        "Applied Delta-u respects common limit = True"
    )

    print(
        "Physical control =",
        u_second_applied
    )

    assert np.all(
        np.isfinite(
            current_u
        )
    )

    assert np.all(
        np.isfinite(
            u_second_applied
        )
    )

    print()
    print(
        "Second MPC step: PASSED"
    )

    # ========================================================
    # 6. Reset Validation
    # ========================================================

    print_section(
        6,
        "Controller Reset"
    )

    controller.reset()

    print(
        "u_prev after reset =",
        controller.u_prev_mpc
    )

    print(
        "Physical control after reset =",
        controller.last_physical_control
    )

    assert np.allclose(
        controller.u_prev,
        np.array(
            [
                controller.hover_thrust,
                0.0,
                0.0,
                0.0
            ],
            dtype=float
        ),
        atol=1e-6
    )

    print(
        "Base physical u_prev after reset =",
        controller.u_prev
    )

    assert np.allclose(
        controller.u_prev_mpc,
        np.zeros(4)
    )

    assert np.allclose(
        controller.last_physical_control,
        np.array(
            [
                controller.hover_thrust,
                0.0,
                0.0,
                0.0
            ]
        ),
        atol=1e-6
    )

    print()
    print(
        "Controller reset: PASSED"
    )

    # ========================================================
    # 7. Infeasible-State Fail-Safe
    # ========================================================

    print_section(
        7,
        "Infeasible-State Fail-Safe"
    )

    # --------------------------------------------------------
    # Start from a known feasible applied command
    # --------------------------------------------------------

    controller.reset()

    x = np.zeros(
        12,
        dtype=float
    )

    x_ref = np.zeros(
        12,
        dtype=float
    )

    x_ref[2] = 1.0

    feasible_requested = (
        controller.control_law(
            x,
            x_ref
        )
    )

    # Emulate the real ControllerBase pipeline:
    #
    #     control_law()
    #         ->
    #     clamp_u()
    #
    # This is essential because clamp_u() synchronizes
    # u_prev_mpc with the ACTUALLY applied command.
    feasible_applied = (
        controller.clamp_u(
            feasible_requested
        )
    )

    previous_mpc_control = (
        controller.u_prev_mpc.copy()
    )

    previous_physical_control = (
        feasible_applied.copy()
    )

    infeasible_count_before = (
        controller.infeasible_count
    )

    print(
        "Previous feasible MPC control =",
        previous_mpc_control
    )

    print(
        "Previous feasible physical control =",
        previous_physical_control
    )

    # --------------------------------------------------------
    # Deliberately extreme altitude state
    #
    # z is far outside the normal operating envelope.
    # With the soft-QP this may lead OSQP to max iterations,
    # but the optimizer must reject a numerically infeasible
    # max-iter candidate.
    # --------------------------------------------------------

    x_bad = np.zeros(
        12,
        dtype=float
    )

    x_bad[2] = 100.0

    fallback_requested = (
        controller.control_law(
            x_bad,
            x_ref
        )
    )

    # Again emulate the actual ControllerBase pipeline.
    fallback_applied = (
        controller.clamp_u(
            fallback_requested
        )
    )

    print(
        "Fail-safe requested physical control =",
        fallback_requested
    )

    print(
        "Fail-safe applied physical control =",
        fallback_applied
    )

    print(
        "MPC u_prev after fail-safe =",
        controller.u_prev_mpc
    )

    print(
        "Infeasible count =",
        controller.infeasible_count
    )

    # --------------------------------------------------------
    # Expected rate-limited fail-safe in MPC coordinates
    #
    # Target:
    #
    #     u_safe = [0, 0, 0, 0]
    #
    # corresponding physically to:
    #
    #     [T_hover, 0, 0, 0]
    #
    # Move toward that target subject to du_max_vec.
    # --------------------------------------------------------

    safe_target_mpc = np.zeros(
        controller.parameters.nu,
        dtype=float
    )

    du_guard = np.asarray(
        controller.parameters.du_max_vec,
        dtype=float
    )

    expected_du = np.clip(
        safe_target_mpc
        - previous_mpc_control,
        -du_guard,
        du_guard
    )

    expected_mpc_control = (
        previous_mpc_control
        + expected_du
    )

    expected_physical_control = (
        expected_mpc_control.copy()
    )

    expected_physical_control[0] += (
        controller.hover_thrust
    )

    print(
        "Expected fail-safe MPC control =",
        expected_mpc_control
    )

    print(
        "Expected fail-safe physical control =",
        expected_physical_control
    )

    actual_du = (
        controller.u_prev_mpc
        - previous_mpc_control
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    failure_detected = (
        controller.infeasible_count
        == infeasible_count_before + 1
    )

    requested_correct = np.allclose(
        fallback_requested,
        expected_physical_control,
        atol=1e-6,
        rtol=0.0
    )

    applied_correct = np.allclose(
        fallback_applied,
        expected_physical_control,
        atol=1e-6,
        rtol=0.0
    )

    memory_synchronized = np.allclose(
        controller.u_prev_mpc,
        expected_mpc_control,
        atol=1e-6,
        rtol=0.0
    )

    rate_limit_respected = np.all(
        np.abs(actual_du)
        <= du_guard + 1e-9
    )

    moving_toward_safe_target = np.all(
        np.abs(controller.u_prev_mpc)
        <= np.abs(previous_mpc_control)
        + 1e-9
    )

    print(
        "QP failure detected =",
        failure_detected
    )

    print(
        "Fail-safe requested control correct =",
        requested_correct
    )

    print(
        "Fail-safe applied control correct =",
        applied_correct
    )

    print(
        "Applied-command memory synchronized =",
        memory_synchronized
    )

    print(
        "Fail-safe Delta-U respects limits =",
        rate_limit_respected
    )

    print(
        "Control moved toward safe target =",
        moving_toward_safe_target
    )

    assert failure_detected
    assert requested_correct
    assert applied_correct
    assert memory_synchronized
    assert rate_limit_respected
    assert moving_toward_safe_target

    print()
    print(
        "Infeasible-state fail-safe: PASSED"
    )

    # ========================================================
    # Final Results
    # ========================================================

    print()
    print("=" * 70)
    print(
        " PHYSICS MPC CONTROLLER TEST RESULTS"
    )
    print("=" * 70)

    print(
        "Hover thrust        =",
        controller.hover_thrust
    )

    print(
        "Final MPC u_prev    =",
        controller.u_prev_mpc
    )

    print(
        "Final physical u    =",
        controller.last_physical_control
    )

    print(
        "Successful solves   =",
        controller.solve_count
    )

    print(
        "Infeasible solves   =",
        controller.infeasible_count
    )

    print()
    print(
        "ALL PHYSICS MPC CONTROLLER TESTS PASSED"
    )

    print("=" * 70)


# ============================================================

if __name__ == "__main__":

    main()