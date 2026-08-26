#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_yaw_equilibrium_test.py

Offline diagnostic for the frozen DD21 PEM model.

Purpose
-------
Check whether the absolute S2-Yaw reference

    psi_ref = +45 deg
    r_ref   = 0

is an equilibrium of the centered PEM model

    x_c[k+1] = A x_c[k] + B u_c[k].

No controller tuning is changed.
No Gazebo/ROS execution is required.

Run from:
    ~/duckiedrone_sim/src/duckiedrone_validation/scripts

Command:
    python -m controllers.mpc.pem_yaw_equilibrium_test
"""

import numpy as np

from .parameters import MPCParameters
from .pem_model import PEMModel


STATE_NAMES = [
    "x", "y", "z",
    "phi", "theta", "psi",
    "vx", "vy", "vz",
    "p", "q", "r",
]

INPUT_NAMES = [
    "T", "tau_phi", "tau_theta", "tau_psi",
]


def fmt_vec(v):
    return np.array2string(
        np.asarray(v),
        precision=10,
        suppress_small=False,
        floatmode="fixed",
    )


def main():

    print("=" * 96)
    print("DD21 PEM — YAW EQUILIBRIUM DIAGNOSTIC")
    print("=" * 96)

    # ------------------------------------------------------------------
    # Load the SAME frozen PEM wrapper used by PEM-MPC.
    # ------------------------------------------------------------------

    parameters = MPCParameters()
    model = PEMModel(parameters)

    A = np.asarray(model.A, dtype=float)
    B = np.asarray(model.B, dtype=float)
    Ts = float(model.Ts)

    print()
    print("[1] FROZEN MODEL")
    print("-" * 96)
    print("A shape =", A.shape)
    print("B shape =", B.shape)
    print("Ts      =", Ts)
    print("rho(A)  =", float(np.max(np.abs(np.linalg.eigvals(A)))))

    # ------------------------------------------------------------------
    # Exact scenario reference:
    # absolute psi = +45 deg.
    #
    # The PEM model is centered at x_op, therefore:
    #
    #     x_c = x_abs - x_op
    #
    # ------------------------------------------------------------------

    psi_ref_abs = np.deg2rad(45.0)

    x_abs_target = np.asarray(model.x_op, dtype=float).copy()
    x_abs_target[5] = psi_ref_abs

    x_target = x_abs_target - np.asarray(model.x_op, dtype=float)

    print()
    print("[2] TARGET STATE")
    print("-" * 96)
    print("absolute psi target [deg] =",
          np.rad2deg(x_abs_target[5]))
    print("PEM x_op psi [rad]        =",
          model.x_op[5])
    print("centered psi target [rad] =",
          x_target[5])
    print("centered psi target [deg] =",
          np.rad2deg(x_target[5]))
    print("centered r target [rad/s] =",
          x_target[11])

    # ------------------------------------------------------------------
    # Pure yaw-shift invariance check.
    #
    # For a model in which an arbitrary constant yaw angle with zero
    # yaw rate is an equilibrium at zero centered input, the psi column
    # of A should ideally be the unit vector e_psi:
    #
    #     A @ (alpha e_psi) = alpha e_psi
    #
    # for any constant alpha.
    # ------------------------------------------------------------------

    e_psi = np.zeros(12)
    e_psi[5] = 1.0

    yaw_shift_defect = A[:, 5] - e_psi

    print()
    print("[3] YAW-SHIFT INVARIANCE")
    print("-" * 96)
    print("A[:, psi] =")
    print(fmt_vec(A[:, 5]))
    print()
    print("A[:, psi] - e_psi =")
    print(fmt_vec(yaw_shift_defect))
    print()
    print("||A[:,psi] - e_psi||_2 =",
          np.linalg.norm(yaw_shift_defect))
    print("max abs defect           =",
          np.max(np.abs(yaw_shift_defect)))

    print()
    print("Non-negligible psi-column couplings:")
    found = False
    for i, value in enumerate(yaw_shift_defect):
        if abs(value) > 1.0e-8:
            found = True
            print(
                "  {:>5s} <- psi : {:+.10e}".format(
                    STATE_NAMES[i],
                    value,
                )
            )
    if not found:
        print("  none above 1e-8")

    # ------------------------------------------------------------------
    # One-step equilibrium residual at zero PEM-centered input.
    #
    # u_c = 0 corresponds to the empirical PEM operating input.
    # ------------------------------------------------------------------

    u_zero = np.zeros(4)

    x_next = A @ x_target + B @ u_zero
    residual = x_next - x_target

    print()
    print("[4] ONE-STEP EQUILIBRIUM RESIDUAL AT u_c = 0")
    print("-" * 96)
    print("||x_next - x_target||_2 =",
          np.linalg.norm(residual))
    print("max abs residual         =",
          np.max(np.abs(residual)))

    print()
    print("State residuals:")
    for i, name in enumerate(STATE_NAMES):
        print(
            "  {:>5s}: {:+.10e}".format(
                name,
                residual[i],
            )
        )

    print()
    print("Yaw-specific residual:")
    print("  delta psi one-step [rad] =",
          residual[5])
    print("  delta psi one-step [deg] =",
          np.rad2deg(residual[5]))
    print("  generated r [rad/s]      =",
          x_next[11])
    print("  generated r [deg/s]      =",
          np.rad2deg(x_next[11]))

    # ------------------------------------------------------------------
    # Constant centered input required by the IDENTIFIED MODEL to hold
    # this target as closely as possible:
    #
    #     x_target = A x_target + B u_eq
    #
    # =>  B u_eq = (I - A) x_target
    #
    # ------------------------------------------------------------------

    rhs = (np.eye(12) - A) @ x_target

    u_eq_all, _, _, _ = np.linalg.lstsq(
        B,
        rhs,
        rcond=None,
    )

    eq_residual_all = (
        A @ x_target
        + B @ u_eq_all
        - x_target
    )

    print()
    print("[5] BEST CONSTANT INPUT FOR 45-deg YAW EQUILIBRIUM")
    print("-" * 96)
    print("Least-squares u_eq in PEM centered coordinates:")
    for i, name in enumerate(INPUT_NAMES):
        print(
            "  {:>9s}: {:+.10e}".format(
                name,
                u_eq_all[i],
            )
        )

    print()
    print("Equilibrium residual with full 4-input u_eq:")
    print("  norm    =",
          np.linalg.norm(eq_residual_all))
    print("  max abs =",
          np.max(np.abs(eq_residual_all)))
    print("  psi     =",
          eq_residual_all[5])
    print("  r       =",
          eq_residual_all[11])

    # ------------------------------------------------------------------
    # Yaw-torque-only equilibrium fit.
    # This isolates how much tau_psi alone can compensate the learned
    # yaw-state defect.
    # ------------------------------------------------------------------

    b_yaw = B[:, 3]

    denom = float(b_yaw @ b_yaw)

    if denom > 0.0:
        tau_psi_only = float(
            (b_yaw @ rhs) / denom
        )
    else:
        tau_psi_only = np.nan

    u_yaw_only = np.zeros(4)
    u_yaw_only[3] = tau_psi_only

    eq_residual_yaw_only = (
        A @ x_target
        + B @ u_yaw_only
        - x_target
    )

    print()
    print("[6] YAW-TORQUE-ONLY EQUILIBRIUM FIT")
    print("-" * 96)
    print("tau_psi required [model coord] =",
          tau_psi_only)
    print("residual norm                  =",
          np.linalg.norm(eq_residual_yaw_only))
    print("residual psi                   =",
          eq_residual_yaw_only[5])
    print("residual r                     =",
          eq_residual_yaw_only[11])

    # ------------------------------------------------------------------
    # 20-step open prediction from exact 45-deg target with zero
    # centered input. This is the MPC prediction horizon (0.2 s).
    # ------------------------------------------------------------------

    print()
    print("[7] ZERO-INPUT 20-STEP PREDICTION FROM 45 deg")
    print("-" * 96)
    print(
        "{:>4s} {:>13s} {:>13s} {:>13s}".format(
            "k",
            "psi [deg]",
            "r [deg/s]",
            "dpsi [deg]",
        )
    )

    xk = x_target.copy()

    for k in range(1, 21):

        xk = A @ xk

        psi_deg = np.rad2deg(
            xk[5] + model.x_op[5]
        )
        r_deg = np.rad2deg(
            xk[11] + model.x_op[11]
        )
        dpsi_deg = (
            psi_deg - 45.0
        )

        print(
            "{:4d} {:13.8f} {:13.8f} {:+13.8f}".format(
                k,
                psi_deg,
                r_deg,
                dpsi_deg,
            )
        )

    print()
    print("20-step terminal:")
    print("  psi [deg]      =",
          np.rad2deg(xk[5] + model.x_op[5]))
    print("  psi error [deg] =",
          np.rad2deg(xk[5] + model.x_op[5]) - 45.0)
    print("  r [deg/s]      =",
          np.rad2deg(xk[11] + model.x_op[11]))

    # ------------------------------------------------------------------
    # Interpretation gate.
    # Do NOT alter the model here.
    # ------------------------------------------------------------------

    equilibrium_tol = 1.0e-8

    pure_yaw_equilibrium = bool(
        np.max(np.abs(residual))
        <= equilibrium_tol
    )

    print()
    print("[8] RESULT")
    print("-" * 96)
    print(
        "45-deg yaw is zero-input PEM equilibrium =",
        pure_yaw_equilibrium,
    )

    if pure_yaw_equilibrium:
        print(
            "The frozen PEM model itself preserves a constant "
            "45-deg yaw at u_c=0 within the test tolerance."
        )
    else:
        print(
            "The frozen PEM model does NOT preserve the exact "
            "45-deg yaw state at u_c=0."
        )
        print(
            "This is a candidate structural source of steady-state "
            "yaw offset in finite-horizon PEM-MPC."
        )

    print()
    print("No model or controller parameters were modified.")
    print("=" * 96)


if __name__ == "__main__":
    main()
