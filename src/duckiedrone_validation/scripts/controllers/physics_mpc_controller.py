#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
physics_mpc_controller.py

Physics-based MPC controller integration
for the Duckiedrone DD21 validation framework.

The controller uses the validated Physics MPC core:

    PhysicsModel
        -> Phi, Gamma
        -> CostFunction
        -> Constraints
        -> Optimizer (OSQP)

MPC state:

    x = [
        x, y, z,
        phi, theta, psi,
        vx, vy, vz,
        p, q, r
    ]

Internal MPC input:

    u_mpc = [
        delta_T,
        tau_phi,
        tau_theta,
        tau_psi
    ]

where:

    delta_T = T - T_hover

The output returned to ControllerBase is the physical wrench:

    u_physical = [
        T,
        tau_phi,
        tau_theta,
        tau_psi
    ]

with:

    T = T_hover + delta_T

Author: Abdallah GHOUL 2026
"""

import copy
import os
import sys

import numpy as np
import rospy
import rospkg


# ============================================================
# Package paths
# ============================================================

_PKG = rospkg.RosPack().get_path(
    "duckiedrone_validation"
)

_CONTROLLERS_PATH = os.path.join(
    _PKG,
    "scripts",
    "controllers"
)

if _CONTROLLERS_PATH not in sys.path:
    sys.path.insert(
        0,
        _CONTROLLERS_PATH
    )


# ============================================================
# Validation Controller Interface
# ============================================================

from controller_base import ControllerBase


# ============================================================
# Physics MPC Core
# ============================================================

from mpc.parameters import MPCParameters
from mpc.vehicle_parameters import VehicleParameters
from mpc.physics_model import PhysicsModel
from mpc.optimizer import Optimizer


# ============================================================
# Physics MPC Controller
# ============================================================

class PhysicsMPCController(ControllerBase):
    """
    Physics-based MPC controller for the Duckiedrone DD21.

    The QP works internally with thrust deviation:

        delta_T = T - T_hover

    while ControllerBase receives the physical wrench:

        [T, tau_phi, tau_theta, tau_psi]
    """

    def __init__(
        self,
        controller_name="physics_mpc_controller",
        prediction_model_class=PhysicsModel,
        state_offset=None,
        input_offset=None
    ):

        # ----------------------------------------------------
        # Base controller
        # ----------------------------------------------------

        super(
            PhysicsMPCController,
            self
        ).__init__(
            controller_name
        )

        self.controller_name = str(
            controller_name
        )

        self.prediction_model_class = (
            prediction_model_class
        )

        # ----------------------------------------------------
        # Timing fix B (2026-08-23):
        # solve the QP only on fresh odometry snapshots.
        # Stale-state solves were the confirmed root cause of
        # the pitch oscillation; PID is unaffected.
        # ----------------------------------------------------

        self.require_fresh_odom = True

        # ----------------------------------------------------
        # One-step input-delay compensation (Fix D, 2026-08-23)
        #
        # The plant applies each command one control period
        # late (measured signature: e_q = -Bd*u in run 7).
        # Set False to A/B-test against the old behavior.
        # ----------------------------------------------------
        self.delay_compensation = True

        # ----------------------------------------------------
        # MPC Parameters
        #
        # Runtime MPC tuning comes from /mpc.
        # Physical and state constraints come from /dd21.
        #
        # This keeps a single source of truth:
        #
        #   /mpc  -> horizons, weights, solver settings
        #   /dd21 -> actuator/state constraints
        #
        # MPCParameters remains a plain data container and keeps
        # its defaults for offline/unit tests.
        # ----------------------------------------------------

        mpc_cfg = rospy.get_param("/mpc")
        dd21_cfg = rospy.get_param("/dd21")

        Np = int(mpc_cfg["Np"])
        Nc = int(mpc_cfg["Nc"])
        Ts = float(mpc_cfg["Ts"])

        # The current prediction/core implementation has been
        # validated for a full control horizon only.
        if Nc != Np:
            raise ValueError(
                "Current Physics MPC core requires Nc == Np. "
                "Move blocking for Nc < Np is not implemented yet. "
                "Received Np={} Nc={}.".format(Np, Nc)
            )

        # ----------------------------------------------------
        # Cost matrices from ROS YAML
        # ----------------------------------------------------

        q_diag = np.asarray(
            mpc_cfg["Q"],
            dtype=float
        )

        r_diag = np.asarray(
            mpc_cfg["R"],
            dtype=float
        )

        s_diag = np.asarray(
            mpc_cfg["S"],
            dtype=float
        )

        if q_diag.shape != (12,):
            raise ValueError(
                "Expected /mpc/Q to contain 12 diagonal weights."
            )

        if r_diag.shape != (4,):
            raise ValueError(
                "Expected /mpc/R to contain 4 diagonal weights."
            )

        if s_diag.shape != (4,):
            raise ValueError(
                "Expected /mpc/S to contain 4 diagonal weights."
            )

        Q = np.diag(q_diag)
        R = np.diag(r_diag)
        S = np.diag(s_diag)

        P_scale = float(
            mpc_cfg.get("P_scale", 1.0)
        )

        P = P_scale * Q

        # ----------------------------------------------------
        # Solver settings supported by the current Optimizer
        # interface
        # ----------------------------------------------------

        osqp_cfg = mpc_cfg.get(
            "osqp",
            {}
        )

        eps_abs = float(
            osqp_cfg.get("eps_abs", 1.0e-5)
        )

        eps_rel = float(
            osqp_cfg.get("eps_rel", 1.0e-5)
        )

        # Optimizer currently exposes one common tolerance.
        solver_tolerance = max(
            eps_abs,
            eps_rel
        )

        # ----------------------------------------------------
        # Prediction-model coordinate offsets
        # ----------------------------------------------------

        if state_offset is None:

            state_offset = np.zeros(
                12,
                dtype=float
            )

        if input_offset is None:

            input_offset = np.zeros(
                4,
                dtype=float
            )

        state_offset = np.asarray(
            state_offset,
            dtype=float
        ).reshape(12)

        input_offset = np.asarray(
            input_offset,
            dtype=float
        ).reshape(4)

        # ----------------------------------------------------
        # Build runtime MPC parameter container
        # ----------------------------------------------------

        self.parameters = MPCParameters(

            # Prediction
            Ts=Ts,
            Np=Np,
            Nc=Nc,
            mass=float(
                dd21_cfg["mass"]
            ),
            gravity=float(
                dd21_cfg["gravity"]
            ),

            Ixx=float(
                dd21_cfg["Ixx"]
            ),
            Iyy=float(
                dd21_cfg["Iyy"]
            ),
            Izz=float(
                dd21_cfg["Izz"]
            ),
            # Prediction-model coordinate system
            state_offset=state_offset,
            input_offset=input_offset,

            # Cost
            Q=Q,
            R=R,
            S=S,
            P=P,

            # Solver
            max_iterations=int(
                osqp_cfg.get(
                    "max_iter",
                    200
                )
            ),
            tolerance=solver_tolerance,

            # State constraints — same source as ControllerBase
            phi_max=float(
                dd21_cfg["phi_max"]
            ),
            theta_max=float(
                dd21_cfg["theta_max"]
            ),

            z_min=float(
                dd21_cfg.get(
                    "z_min",
                    0.0
                )
            ),
            z_max=float(
                dd21_cfg["z_max"]
            ),

            # Physical actuator constraints
            thrust_min=float(
                dd21_cfg["T_min"]
            ),
            thrust_max=float(
                dd21_cfg["T_max"]
            ),

            torque_max=float(
                dd21_cfg["tau_max"]
            ),

            du_max=float(
                dd21_cfg["du_max"]
            ),

            # Per-channel increment bounds (Fix E,
            # 2026-08-24): the shared scalar du_max=5.0 is
            # 5x the full torque range, so the QP Delta-U
            # constraint was structurally present but
            # numerically inactive. Torque channels are
            # rate-limited to 0.05 N.m per 10 ms sample
            # via /dd21/du_max_vec; PID is untouched.
            du_max_vec=np.asarray(
                dd21_cfg.get(
                    "du_max_vec",
                    [
                        float(dd21_cfg["du_max"]),
                        0.05,
                        0.05,
                        0.05
                    ]
                ),
                dtype=float
            ),

            # Rotor allocation / actuator feasibility
            k_f=float(
                dd21_cfg["k_f"]
            ),
            k_m=float(
                dd21_cfg["k_m"]
            ),
            arm_dx=float(
                dd21_cfg.get(
                    "arm_dx",
                    0.0775
                )
            ),
            arm_dy=float(
                dd21_cfg.get(
                    "arm_dy",
                    0.1075
                )
            ),
            max_rotor_velocity=float(
                dd21_cfg["max_rotor_velocity"]
            ),
            # Soft state constraints
            soft_state_constraints=bool(
                mpc_cfg.get(
                    "soft_state_constraints",
                    True
                )
            ),

            slack_weight=float(
                mpc_cfg.get(
                    "slack_weight",
                    1.0e4
                )
            )
        )

        # ----------------------------------------------------
        # Vehicle Parameters
        # ----------------------------------------------------

        self.vehicle = (
            VehicleParameters()
        )

        self.hover_thrust = (
            self.parameters.mass
            * self.parameters.gravity
        )

        # ----------------------------------------------------
        # Physics Prediction Model
        # ----------------------------------------------------

        self.model = (
            self.prediction_model_class(
                self.parameters
            )
        )

        self.Phi = (
            self.model.build_phi()
        )

        self.Gamma = (
            self.model.build_gamma()
        )

        # ----------------------------------------------------
        # QP Optimizer
        # ----------------------------------------------------

        self.optimizer = Optimizer(
            self.parameters,
            self.Phi,
            self.Gamma
        )

        # ----------------------------------------------------
        # S2 Roll/Pitch attitude-only weighting
        #
        # During the direct attitude-reference interval:
        #
        #   Qx  = 0
        #   Qy  = 0
        #   Qvx = 0
        #   Qvy = 0
        #
        # State order:
        # [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        #
        # All other weights, constraints, R and S remain
        # identical to the nominal Physics MPC.
        # ----------------------------------------------------

        self.parameters_s2_attitude = copy.deepcopy(
            self.parameters
        )

        Q_s2_attitude = (
            self.parameters.Q.copy()
        )

        # x, y, vx, vy
        Q_s2_attitude[0, 0] = 0.0
        Q_s2_attitude[1, 1] = 0.0
        Q_s2_attitude[6, 6] = 0.0
        Q_s2_attitude[7, 7] = 0.0

        self.parameters_s2_attitude.Q = (
            Q_s2_attitude
        )

        # Terminal penalty must use the same S2 weighting.
        # Otherwise x/y/vx/vy would still be penalized at
        # the end of the prediction horizon.
        self.parameters_s2_attitude.P = (
            P_scale * Q_s2_attitude
        )

        self.optimizer_s2_attitude = Optimizer(
            self.parameters_s2_attitude,
            self.Phi,
            self.Gamma
        )

        # ----------------------------------------------------
        # Previous MPC input
        #
        # IMPORTANT:
        #
        # Stored in deviation coordinates:
        #
        # [
        #   delta_T,
        #   tau_phi,
        #   tau_theta,
        #   tau_psi
        # ]
        #
        # delta_T = 0 means hover thrust.
        # ----------------------------------------------------

        # Previous ACTUALLY applied input expressed in
        # prediction-model coordinates.
        #
        # The physical startup command is nominal hover:
        #
        #     [T_hover, 0, 0, 0]
        #
        # Physics:
        #     -> [0, 0, 0, 0]
        #
        # PEM:
        #     -> -input_offset
        self.u_prev_mpc = (
            self._nominal_to_model_input(
                np.zeros(
                    self.parameters.nu,
                    dtype=float
                )
            )
        )

        # ----------------------------------------------------
        # Last physical command
        # ----------------------------------------------------

        self.last_physical_control = np.array(
            [
                self.hover_thrust,
                0.0,
                0.0,
                0.0
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        self.solve_count = 0
        self.infeasible_count = 0

        self.last_solver_info = None

        # ----------------------------------------------------
        # Closed-loop causality diagnostics
        #
        # Diagnostic only:
        # does NOT modify the MPC solution or applied command.
        # ----------------------------------------------------

        self._diag_x = None
        self._diag_x_ref = None

        self._diag_u_requested_mpc = None
        self._diag_u_requested_physical = None

        self._diag_solver_status = "unknown"
        self._diag_slack_max = 0.0

        # Keep an independent memory of the previously APPLIED
        # physical wrench.  Do not use self.u_prev_mpc here,
        # because control_law() temporarily stores the optimizer
        # request there before clamp_u() synchronizes it.
        self._diag_prev_applied_physical = np.array(
            [
                self.hover_thrust,
                0.0,
                0.0,
                0.0
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # One-step model/plant consistency diagnostic
        # ----------------------------------------------------

        self._modelcheck_prev_x = None
        self._modelcheck_prev_u_mpc = None
        self._modelcheck_prev_t = None

        self._diag_state_time = None

        # ----------------------------------------------------
        # v3 attitude envelope guard (recovery mode)
        #
        # The linear prediction model is only valid near the
        # hover attitude. If phi/theta or the body rates leave
        # that envelope (e.g. solver stalls, gusts, aggressive
        # references), the QP is bypassed and a simple
        # independent PD leveling law drives the attitude back.
        #
        # Entry and exit are hysteretic. All recovery commands
        # still pass through clamp_u(), so the per-channel slew
        # guard remains active during recovery.
        # ----------------------------------------------------

        self.recovery_enabled = bool(
            dd21_cfg.get("recovery_enabled", True)
        )

        # Entry envelope (rad / rad per second).
        self.recovery_att_enter = float(
            dd21_cfg.get("recovery_att_enter", 0.40)
        )

        self.recovery_rate_enter = float(
            dd21_cfg.get("recovery_rate_enter", 2.5)
        )

        # Exit envelope (hysteresis).
        self.recovery_att_exit = float(
            dd21_cfg.get("recovery_att_exit", 0.12)
        )

        self.recovery_rate_exit = float(
            dd21_cfg.get("recovery_rate_exit", 0.8)
        )

        # PD leveling gains (N.m per rad / per rad/s).
        self.recovery_k_att = float(
            dd21_cfg.get("recovery_k_att", 0.25)
        )

        self.recovery_k_rate = float(
            dd21_cfg.get("recovery_k_rate", 0.05)
        )

        # Torque cap during recovery (N.m).
        self.recovery_torque_cap = float(
            dd21_cfg.get("recovery_torque_cap", 0.15)
        )

        # If recovery lasts longer than this [s], start a
        # gentle descent while leveling (delta_T below hover).
        self.recovery_max_time = float(
            dd21_cfg.get("recovery_max_time", 3.0)
        )

        self.recovery_descent_delta_t = float(
            dd21_cfg.get("recovery_descent_delta_t", -0.5)
        )

        self._recovery_active = False

        self._recovery_since = None

        self.recovery_count = 0

        # ----------------------------------------------------
        # Startup information
        # ----------------------------------------------------

        rospy.loginfo(
            "=== Physics MPC Controller initialized ==="
        )

        rospy.loginfo(
            "Hover thrust: %.6f N",
            self.hover_thrust
        )

        rospy.loginfo(
            "MPC dimensions: nx=%d nu=%d Np=%d Nc=%d",
            self.parameters.nx,
            self.parameters.nu,
            self.parameters.Np,
            self.parameters.Nc
        )

    # ========================================================

    def _validate_state(
        self,
        x
    ):
        """
        Validate and format the current state.
        """

        x = np.asarray(
            x,
            dtype=float
        ).reshape(-1)

        if x.size != self.parameters.nx:

            raise ValueError(
                "Physics MPC state must contain "
                f"{self.parameters.nx} elements, "
                f"got {x.size}."
            )

        if not np.all(
            np.isfinite(x)
        ):

            raise ValueError(
                "Physics MPC state contains "
                "non-finite values."
            )

        return x

    # ========================================================

    def _validate_reference(
        self,
        x_ref
    ):
        """
        Validate the reference state.

        Supported:

            x_ref.shape == (12,)

        or a complete stacked trajectory:

            x_ref.shape == (Np*nx,)
        """

        x_ref = np.asarray(
            x_ref,
            dtype=float
        ).reshape(-1)

        valid_sizes = (
            self.parameters.nx,
            self.parameters.Np
            * self.parameters.nx
        )

        if x_ref.size not in valid_sizes:

            raise ValueError(
                "Physics MPC reference must have "
                "size nx or Np*nx."
            )

        if not np.all(
            np.isfinite(x_ref)
        ):

            raise ValueError(
                "Physics MPC reference contains "
                "non-finite values."
            )

        return x_ref

    # ========================================================

    def _state_to_model_coordinates(
        self,
        x_absolute
    ):
        """
        Convert an absolute physical state to the coordinate
        system used by the active prediction model.

        Convention:

            x_model =
                x_absolute - state_offset

        Physics MPC:
            state_offset = 0

        PEM-MPC:
            state_offset = x_op
        """

        x_absolute = np.asarray(
            x_absolute,
            dtype=float
        ).reshape(
            self.parameters.nx
        )

        return (
            x_absolute
            - self.parameters.state_offset
        )

    # ========================================================

    def _state_to_absolute_coordinates(
        self,
        x_model
    ):
        """
        Convert prediction-model state coordinates back to
        absolute physical state coordinates.
        """

        x_model = np.asarray(
            x_model,
            dtype=float
        ).reshape(
            self.parameters.nx
        )

        return (
            x_model
            + self.parameters.state_offset
        )

    # ========================================================

    def _reference_to_model_coordinates(
        self,
        x_ref_absolute
    ):
        """
        Convert either:

            (nx,)

        or:

            (Np*nx,)

        absolute reference coordinates to prediction-model
        coordinates.
        """

        x_ref_absolute = np.asarray(
            x_ref_absolute,
            dtype=float
        ).reshape(-1)

        nx = self.parameters.nx

        if x_ref_absolute.size == nx:

            return (
                x_ref_absolute
                - self.parameters.state_offset
            )

        if (
            x_ref_absolute.size
            == self.parameters.Np * nx
        ):

            ref_matrix = (
                x_ref_absolute.reshape(
                    self.parameters.Np,
                    nx
                )
            )

            ref_model = (
                ref_matrix
                - self.parameters.state_offset[
                    None,
                    :
                ]
            )

            return ref_model.reshape(-1)

        raise ValueError(
            "Reference size is incompatible with "
            "prediction-model coordinate conversion."
        )

    # ========================================================

    def _nominal_to_model_input(
        self,
        u_nominal
    ):
        """
        Convert nominal hover-deviation input to the active
        prediction-model input coordinates.

        Convention:

            u_nominal =
                [T-T_hover,
                 tau_phi,
                 tau_theta,
                 tau_psi]

            u_model =
                u_nominal - input_offset

        Physics MPC:
            input_offset = 0

        PEM-MPC:
            input_offset = u_trim_nominal
        """

        u_nominal = np.asarray(
            u_nominal,
            dtype=float
        ).reshape(
            self.parameters.nu
        )

        return (
            u_nominal
            - self.parameters.input_offset
        )

    # ========================================================

    def _model_to_nominal_input(
        self,
        u_model
    ):
        """
        Convert active prediction-model input to nominal
        hover-deviation coordinates.
        """

        u_model = np.asarray(
            u_model,
            dtype=float
        ).reshape(
            self.parameters.nu
        )

        return (
            u_model
            + self.parameters.input_offset
        )

    # ========================================================

    def _physical_to_model_input(
        self,
        u_physical
    ):
        """
        Convert physical wrench:

            [T, tau_phi, tau_theta, tau_psi]

        directly to prediction-model coordinates.
        """

        u_physical = np.asarray(
            u_physical,
            dtype=float
        ).reshape(
            self.parameters.nu
        )

        u_nominal = (
            u_physical.copy()
        )

        u_nominal[0] -= (
            self.hover_thrust
        )

        return (
            self._nominal_to_model_input(
                u_nominal
            )
        )

    # ========================================================

    def _model_to_physical_input(
        self,
        u_model
    ):
        """
        Convert prediction-model input directly to physical
        wrench coordinates.
        """

        u_nominal = (
            self._model_to_nominal_input(
                u_model
            )
        )

        u_physical = (
            u_nominal.copy()
        )

        u_physical[0] += (
            self.hover_thrust
        )

        return u_physical

    # ========================================================

    def _to_physical_control(
        self,
        u_mpc
    ):
        """
        Convert MPC deviation input to physical wrench.

        MPC:

            [
                delta_T,
                tau_phi,
                tau_theta,
                tau_psi
            ]

        Physical:

            [
                T_hover + delta_T,
                tau_phi,
                tau_theta,
                tau_psi
            ]
        """

        u_mpc = np.asarray(
            u_mpc,
            dtype=float
        ).reshape(
            self.parameters.nu
        )

        u_physical = (
            self._model_to_physical_input(
                u_mpc
            )
        )

        # ----------------------------------------------------
        # Safety clipping
        #
        # The QP already enforces these limits.
        # This clipping only protects the controller interface
        # against numerical round-off.
        # ----------------------------------------------------

        u_physical[0] = np.clip(
            u_physical[0],
            self.parameters.thrust_min,
            self.parameters.thrust_max
        )

        u_physical[1:4] = np.clip(
            u_physical[1:4],
            -self.parameters.torque_max,
            self.parameters.torque_max
        )

        return u_physical

    def clamp_u(self, u):
        """
        Apply the common ControllerBase physical safety limits and
        synchronize the MPC previous-input memory with the command
        that is actually sent to the plant.

        ControllerBase works in physical coordinates:

            [T, tau_phi, tau_theta, tau_psi]

        Physics MPC stores its previous input in deviation coordinates:

            [delta_T, tau_phi, tau_theta, tau_psi]

        Therefore:

            delta_T = T_applied - T_hover
        """

        # ----------------------------------------------------
        # Apply the exact same physical clamp used by all
        # controllers.
        # ----------------------------------------------------

        u_applied = super(
            PhysicsMPCController,
            self
        ).clamp_u(
            np.asarray(
                u,
                dtype=float
            ).copy()
        )

        # ----------------------------------------------------
        # v2 post-hoc per-channel slew guard
        #
        # Hard safety net independent of the QP: the applied
        # physical command may never change faster than
        # parameters.du_max_vec per control cycle, no matter
        # what the optimizer or the base-class limiter did.
        #
        # Thrust is in physical coordinates here, but the
        # hover offset cancels inside a difference, so the
        # same du_max_vec applies channel-by-channel.
        # ----------------------------------------------------

        du_guard = np.asarray(
            self.parameters.du_max_vec,
            dtype=float
        ).reshape(
            self.parameters.nu
        )

        u_applied = (
            self._diag_prev_applied_physical
            + np.clip(
                u_applied
                - self._diag_prev_applied_physical,
                -du_guard,
                du_guard
            )
        )

        # ----------------------------------------------------
        # Closed-loop torque causality diagnostic
        # ----------------------------------------------------

        previous_applied = (
            self._diag_prev_applied_physical.copy()
        )

        delta_applied = (
            u_applied - previous_applied
        )

        if (
            self._diag_x is not None
            and self._diag_x_ref is not None
            and self._diag_u_requested_physical is not None
        ):

            x_diag = self._diag_x
            xr_diag = self._diag_x_ref
            u_req = self._diag_u_requested_physical

            # Periodic closed-loop trace.
            rospy.loginfo_throttle(
                0.5,
                "MPC TRACE: "
                "t=%.3f "
                "z=%.4f zr=%.4f "
                "phi=%.4f phir=%.4f "
                "theta=%.4f thetar=%.4f "
                "p=%.4f q=%.4f "
                "REQ=[%.4f %.4f %.4f %.4f] "
                "APP=[%.4f %.4f %.4f %.4f] "
                "DTAU=[%.4f %.4f] "
                "slack=%.5f status=%s",
                rospy.get_time(),
                x_diag[2],
                xr_diag[2],
                x_diag[3],
                xr_diag[3],
                x_diag[4],
                xr_diag[4],
                x_diag[9],
                x_diag[10],
                u_req[0],
                u_req[1],
                u_req[2],
                u_req[3],
                u_applied[0],
                u_applied[1],
                u_applied[2],
                u_applied[3],
                delta_applied[1],
                delta_applied[2],
                self._diag_slack_max,
                self._diag_solver_status
            )

            # Event diagnostic:
            # capture large single-cycle torque changes even if
            # they occur between two throttled TRACE messages.
            if (
                abs(delta_applied[1]) > 0.20
                or abs(delta_applied[2]) > 0.20
            ):

                rospy.logwarn(
                    "MPC TORQUE JUMP: "
                    "t=%.3f "
                    "phi=%.4f theta=%.4f "
                    "p=%.4f q=%.4f "
                    "tau_prev=[%.4f %.4f] "
                    "tau_req=[%.4f %.4f] "
                    "tau_app=[%.4f %.4f] "
                    "dtau=[%.4f %.4f]",
                    rospy.get_time(),
                    x_diag[3],
                    x_diag[4],
                    x_diag[9],
                    x_diag[10],
                    previous_applied[1],
                    previous_applied[2],
                    u_req[1],
                    u_req[2],
                    u_applied[1],
                    u_applied[2],
                    delta_applied[1],
                    delta_applied[2]
                )

        # Update diagnostic memory only AFTER delta computation.
        self._diag_prev_applied_physical = (
            u_applied.copy()
        )

        # ----------------------------------------------------
        # Synchronize MPC memory with the ACTUALLY APPLIED
        # command, not merely the optimizer request.
        # ----------------------------------------------------

        self.u_prev_mpc = (
            self._physical_to_model_input(
                u_applied
            )
        )

        # ----------------------------------------------------
        # Keep physical diagnostic memory synchronized too.
        # ----------------------------------------------------

        self.last_physical_control = (
            u_applied.copy()
        )

        # ----------------------------------------------------
        # Store state + ACTUALLY applied input for prediction
        # of the next measured state.
        # ----------------------------------------------------

        if (
            self._diag_x is not None
            and self._diag_state_time is not None
        ):

            u_applied_mpc = (
                self._physical_to_model_input(
                    u_applied
                )
            )

            self._modelcheck_prev_x = (
                self._state_to_model_coordinates(
                    self._diag_x
                )
            )

            self._modelcheck_prev_u_mpc = (
                u_applied_mpc
            )

            self._modelcheck_prev_t = (
                self._diag_state_time
            )

        return u_applied

    # ========================================================

    def control_law(
        self,
        x,
        x_ref
    ):
        """
        Compute one Physics MPC control action.

        Parameters
        ----------
        x : array-like
            Current 12-state vector.

        x_ref : array-like
            Desired state or stacked reference.

        Returns
        -------
        np.ndarray, shape (4,)

            Physical wrench:

            [
                thrust,
                tau_phi,
                tau_theta,
                tau_psi
            ]
        """

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        x = self._validate_state(
            x
        )

        x_ref = self._validate_reference(
            x_ref
        )

        # ----------------------------------------------------
        # Prediction-model coordinates
        #
        # Keep x and x_ref above in ABSOLUTE coordinates for:
        #
        #   recovery
        #   diagnostics
        #   scenario interpretation
        #
        # Only the model/QP uses the centered coordinates.
        # ----------------------------------------------------

        x_model = (
            self._state_to_model_coordinates(
                x
            )
        )

        x_ref_model = (
            self._reference_to_model_coordinates(
                x_ref
            )
        )

        # ----------------------------------------------------
        # Diagnostic defaults for this control cycle
        # ----------------------------------------------------

        diag_solver_status = "failed"
        diag_slack_max = float("nan")

        # ----------------------------------------------------
        # One-step model/plant consistency check
        # ----------------------------------------------------

        current_state_time = rospy.get_time()

        if (
            self._modelcheck_prev_x is not None
            and self._modelcheck_prev_u_mpc is not None
            and self._modelcheck_prev_t is not None
        ):

            x_pred_model = (
                self.model.Ad.dot(
                    self._modelcheck_prev_x
                )
                + self.model.Bd.dot(
                    self._modelcheck_prev_u_mpc
                )
            )

            # Diagnostic output remains in absolute
            # physical coordinates.
            x_pred = (
                self._state_to_absolute_coordinates(
                    x_pred_model
                )
            )

            dt_observed = (
                current_state_time
                - self._modelcheck_prev_t
            )

            theta_pred = x_pred[4]
            q_pred = x_pred[10]

            theta_meas = x[4]
            q_meas = x[10]

            rospy.loginfo_throttle(
                0.5,
                "MPC MODEL CHECK: "
                "dt=%.5f "
                "theta0=%.5f q0=%.5f "
                "tau0=%.5f "
                "theta_pred=%.5f theta_meas=%.5f "
                "q_pred=%.5f q_meas=%.5f "
                "e_theta=%+.5e e_q=%+.5e",
                dt_observed,
                self._modelcheck_prev_x[4],
                self._modelcheck_prev_x[10],
                self._modelcheck_prev_u_mpc[2],
                theta_pred,
                theta_meas,
                q_pred,
                q_meas,
                theta_meas - theta_pred,
                q_meas - q_pred
            )

        # State timestamp corresponding to the command that
        # will be computed during this cycle.
        self._diag_state_time = current_state_time

        # ----------------------------------------------------
        # v3 attitude envelope guard
        #
        # The linear prediction model is valid only near the
        # hover attitude. While phi/theta or the body rates
        # are outside that envelope, bypass the QP entirely
        # and level the attitude with an independent PD law.
        # The output still passes through clamp_u(), so the
        # per-channel slew guard remains active in recovery.
        # ----------------------------------------------------

        att_now = max(
            abs(x[3]),
            abs(x[4])
        )

        rate_now = max(
            abs(x[9]),
            abs(x[10])
        )

        if self.recovery_enabled:

            if not self._recovery_active:

                if (
                    att_now > self.recovery_att_enter
                    or rate_now > self.recovery_rate_enter
                ):

                    self._recovery_active = True

                    self._recovery_since = (
                        current_state_time
                    )

                    self.recovery_count += 1

                    rospy.logwarn(
                        "Physics MPC RECOVERY activated "
                        "(#%d): phi=%.3f theta=%.3f "
                        "p=%.3f q=%.3f - QP bypassed, "
                        "PD leveling active.",
                        self.recovery_count,
                        x[3],
                        x[4],
                        x[9],
                        x[10]
                    )

            else:

                if (
                    att_now < self.recovery_att_exit
                    and rate_now < self.recovery_rate_exit
                ):

                    self._recovery_active = False

                    self._recovery_since = None

                    self.optimizer.reset_warm_start()
                    self.optimizer_s2_attitude.reset_warm_start()

                    rospy.logwarn(
                        "Physics MPC RECOVERY released: "
                        "phi=%.3f theta=%.3f "
                        "p=%.3f q=%.3f - QP resumed.",
                        x[3],
                        x[4],
                        x[9],
                        x[10]
                    )

        if self._recovery_active:

            # ----------------------------------------------
            # PD leveling law (QP bypassed)
            #
            # Positive tau_phi raises phi (verified against
            # the linear model), so the leveling torque is
            # negative for a positive attitude error.
            # ----------------------------------------------

            recovery_time = (
                current_state_time
                - self._recovery_since
            )

            delta_t_rec = 0.0

            if recovery_time > self.recovery_max_time:

                # Leveling did not succeed in time:
                # descend gently while staying level.
                delta_t_rec = (
                    self.recovery_descent_delta_t
                )

            tau_phi_rec = -(
                self.recovery_k_att * x[3]
                + self.recovery_k_rate * x[9]
            )

            tau_theta_rec = -(
                self.recovery_k_att * x[4]
                + self.recovery_k_rate * x[10]
            )

            u_recovery_nominal = np.array(
                [
                    delta_t_rec,
                    np.clip(
                        tau_phi_rec,
                        -self.recovery_torque_cap,
                        self.recovery_torque_cap
                    ),
                    np.clip(
                        tau_theta_rec,
                        -self.recovery_torque_cap,
                        self.recovery_torque_cap
                    ),
                    0.0
                ],
                dtype=float
            )

            u_mpc = (
                self._nominal_to_model_input(
                    u_recovery_nominal
                )
            )

            diag_solver_status = "recovery"

            diag_slack_max = float("nan")

            rospy.logwarn_throttle(
                1.0,
                "Physics MPC RECOVERY active: "
                "phi=%.3f theta=%.3f "
                "p=%.3f q=%.3f "
                "tau=[%+.4f %+.4f] "
                "t_rec=%.2f",
                x[3],
                x[4],
                x[9],
                x[10],
                u_mpc[1],
                u_mpc[2],
                recovery_time
            )

            u_physical = (
                self._to_physical_control(
                    u_mpc
                )
            )

            self._diag_x = x.copy()

            self._diag_x_ref = (
                x_ref[:self.parameters.nx].copy()
            )

            self._diag_u_requested_mpc = (
                u_mpc.copy()
            )

            self._diag_u_requested_physical = (
                u_physical.copy()
            )

            self._diag_solver_status = (
                diag_solver_status
            )

            self._diag_slack_max = (
                diag_slack_max
            )

            self.last_physical_control = (
                u_physical.copy()
            )

            return u_physical

        # ----------------------------------------------------
        # Solve QP
        # ----------------------------------------------------

        # ----------------------------------------------------
        # One-step input-delay compensation (Fix D)
        #
        # With a one-period actuator delay, the next state is
        # already determined by the APPLIED input:
        #     x_{k+1} = Ad*x_k + Bd*u_{k-1}
        # Solve the QP from that predicted state so the
        # optimizer "sees" the delay. Diagnostics keep using
        # the measured state x (untouched).
        # ----------------------------------------------------
        # ----------------------------------------------------
        # One-step input-delay compensation (Fix D)
        # ----------------------------------------------------

        if self.delay_compensation:

            x_sol = (
                self.model.Ad.dot(
                    x_model
                )
                + self.model.Bd.dot(
                    self.u_prev_mpc
                )
            )

        else:

            x_sol = (
                x_model
            )

        # ----------------------------------------------------
        # Select active MPC optimizer
        #
        # Default:
        #     nominal Physics MPC
        #
        # S2 Roll/Pitch:
        #     attitude-only weighting
        # ----------------------------------------------------

        optimizer_active = self.optimizer

        x_ref_now = x_ref[
            :self.parameters.nx
        ]

        # ----------------------------------------------------
        # Direct attitude-reference detection
        #
        # Use ABSOLUTE commanded attitude, not centered/model
        # coordinates.  A small tolerance avoids interpreting
        # identification trim / numerical noise as an S2
        # attitude command.
        #
        # 1e-3 rad ~= 0.057 deg
        # ----------------------------------------------------

        attitude_reference_threshold = 1.0e-3

        roll_ref_active = (
            abs(x_ref_now[3])
            > attitude_reference_threshold
        )

        pitch_ref_active = (
            abs(x_ref_now[4])
            > attitude_reference_threshold
        )

        attitude_only_reference = (
            roll_ref_active
            or pitch_ref_active
        )

        if attitude_only_reference:

            optimizer_active = (
                self.optimizer_s2_attitude
            )

            rospy.loginfo_throttle(
                1.0,
                "Physics MPC S2 attitude-only weighting active: "
                "Qx=Qy=Qvx=Qvy=0."
            )


        try:

            (
                u_mpc,
                U_opt,
                info
            ) = optimizer_active.solve(
                x_sol,
                x_ref_model,
                self.u_prev_mpc
            )

            # ------------------------------------------------
            # Successful solution
            # ------------------------------------------------

            self.solve_count += 1

            self.last_solver_info = (
                info
            )

            # ------------------------------------------------
            # Runtime Soft-QP diagnostics
            # ------------------------------------------------

            solver_status = str(
                info.get(
                    "status",
                    "unknown"
                )
            )

            solver_iterations = int(
                info.get(
                    "iterations",
                    0
                )
            )

            solver_time_ms = (
                1000.0
                * float(
                    info.get(
                        "solve_time",
                        0.0
                    )
                )
            )

            slack_max = float(
                info.get(
                    "slack_max",
                    0.0
                )
            )

            slack_norm = float(
                info.get(
                    "slack_norm",
                    0.0
                )
            )

            slack_active = bool(
                info.get(
                    "slack_active",
                    False
                )
            )

            diag_solver_status = solver_status
            diag_slack_max = slack_max

            # Periodic solver-health diagnostic
            rospy.loginfo_throttle(
                1.0,
                "Physics MPC solver: "
                "status=%s iter=%d "
                "solve=%.3f ms "
                "slack_max=%.6f "
                "slack_norm=%.6f "
                "slack_active=%s",
                solver_status,
                solver_iterations,
                solver_time_ms,
                slack_max,
                slack_norm,
                str(slack_active)
            )

            # Report active soft-state relaxation.
            #
            # This is especially important for the pitch
            # condition that previously caused a hard-QP
            # infeasibility.
            if slack_active:

                rospy.logwarn_throttle(
                    0.5,
                    "Physics MPC soft constraint active: "
                    "z=%.6f phi=%.6f theta=%.6f "
                    "slack_max=%.6f "
                    "slack_norm=%.6f "
                    "status=%s iter=%d",
                    x[2],
                    x[3],
                    x[4],
                    slack_max,
                    slack_norm,
                    solver_status,
                    solver_iterations
                )

            # solved inaccurate is accepted by Optimizer,
            # but keep it visible during closed-loop testing.
            if (
                solver_status.lower()
                == "solved inaccurate"
            ):

                rospy.logwarn_throttle(
                    0.5,
                    "Physics MPC solver returned "
                    "'solved inaccurate': "
                    "iter=%d solve=%.3f ms "
                    "primal=%.3e dual=%.3e",
                    solver_iterations,
                    solver_time_ms,
                    float(
                        info.get(
                            "primal_residual",
                            0.0
                        )
                    ),
                    float(
                        info.get(
                            "dual_residual",
                            0.0
                        )
                    )
                )

            # v2: maximum-iterations solutions are accepted
            # by the optimizer; keep them visible in the log.
            if bool(
                info.get(
                    "max_iter_hit",
                    False
                )
            ):

                rospy.logwarn_throttle(
                    0.5,
                    "Physics MPC solver hit max iterations "
                    "(accepted): "
                    "iter=%d solve=%.3f ms "
                    "primal=%.3e dual=%.3e",
                    solver_iterations,
                    solver_time_ms,
                    float(
                        info.get(
                            "primal_residual",
                            0.0
                        )
                    ),
                    float(
                        info.get(
                            "dual_residual",
                            0.0
                        )
                    )
                )

            # Previous control MUST remain
            # in MPC deviation coordinates.
            self.u_prev_mpc = (
                u_mpc.copy()
            )

        except RuntimeError as error:

            # ------------------------------------------------
            # Infeasible / failed QP
            #
            # Fail-safe (Fix F, 2026-08-24): apply hover
            # thrust with zero torques instead of holding
            # the previous torque command, which was
            # catastrophic during attitude excursions.
            # ------------------------------------------------

            self.infeasible_count += 1

            # ------------------------------------------------
            # Diagnostic state at infeasible QP
            #
            # State ordering:
            # [x, y, z, phi, theta, psi,
            #  vx, vy, vz, p, q, r]
            # ------------------------------------------------

            if self.infeasible_count <= 5:
                rospy.logwarn(
                    "MPC infeasible diagnostic #%d: "
                    "z=%.6f phi=%.6f theta=%.6f "
                    "vz=%.6f p=%.6f q=%.6f "
                    "u_prev_mpc=[%.6f %.6f %.6f %.6f]",
                    self.infeasible_count,
                    x[2],
                    x[3],
                    x[4],
                    x[8],
                    x[9],
                    x[10],
                    self.u_prev_mpc[0],
                    self.u_prev_mpc[1],
                    self.u_prev_mpc[2],
                    self.u_prev_mpc[3]
                )
                
            rospy.logwarn_throttle(
                1.0,
                "Physics MPC solve failed "
                "(count=%d): %s. "
                "Fail-safe: gliding to hover thrust, "
                "zero torques.",
                self.infeasible_count,
                str(error)
            )

            # Physical fail-safe target remains:
            #
            #     [T_hover, 0, 0, 0]
            #
            # Express that target in the active model coordinates.
            u_hover_mpc = (
                self._nominal_to_model_input(
                    np.zeros(
                        self.parameters.nu,
                        dtype=float
                    )
                )
            )

            du_safe = np.clip(
                u_hover_mpc - self.u_prev_mpc,
                -self.parameters.du_max_vec,
                self.parameters.du_max_vec
            )

            u_mpc = (
                self.u_prev_mpc + du_safe
            )

        # ----------------------------------------------------
        # Convert to physical wrench
        # ----------------------------------------------------

        u_physical = (
            self._to_physical_control(
                u_mpc
            )
        )

        # ----------------------------------------------------
        # Store requested command and corresponding state.
        #
        # The actually applied command is only known later,
        # inside clamp_u().
        # ----------------------------------------------------

        self._diag_x = x.copy()

        # For a stacked reference trajectory, the first nx
        # elements correspond to the first prediction step.
        self._diag_x_ref = (
            x_ref[:self.parameters.nx].copy()
        )

        self._diag_u_requested_mpc = (
            u_mpc.copy()
        )

        self._diag_u_requested_physical = (
            u_physical.copy()
        )

        self._diag_solver_status = (
            diag_solver_status
        )

        self._diag_slack_max = (
            diag_slack_max
        )

        self.last_physical_control = (
            u_physical.copy()
        )

        return u_physical

    # ========================================================

    def reset(self):
        """
        Reset Physics MPC internal state.
        """

        # ----------------------------------------------------
        # Hover-equilibrium MPC coordinates
        # ----------------------------------------------------

        # MPC coordinates:
        # [delta_T, tau_phi, tau_theta, tau_psi]
        self.u_prev_mpc[:] = (
            self._nominal_to_model_input(
                np.zeros(
                    self.parameters.nu,
                    dtype=float
                )
            )
        )

        # ControllerBase coordinates:
        # [T, tau_phi, tau_theta, tau_psi]
        self.u_prev[:] = np.array(
            [
                self.hover_thrust,
                0.0,
                0.0,
                0.0
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # Reset OSQP warm start
        # ----------------------------------------------------

        self.optimizer.reset_warm_start()
        self.optimizer_s2_attitude.reset_warm_start()

        # ----------------------------------------------------
        # Physical output corresponding to delta_T = 0
        # ----------------------------------------------------

        self.last_physical_control[:] = np.array(
            [
                self.hover_thrust,
                0.0,
                0.0,
                0.0
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        self.solve_count = 0
        self.infeasible_count = 0

        self.last_solver_info = None

        self._diag_x = None
        self._diag_x_ref = None

        self._diag_u_requested_mpc = None
        self._diag_u_requested_physical = None

        self._diag_solver_status = "unknown"
        self._diag_slack_max = 0.0

        self._diag_prev_applied_physical[:] = np.array(
            [
                self.hover_thrust,
                0.0,
                0.0,
                0.0
            ],
            dtype=float
        )

        rospy.loginfo(
            "Physics MPC controller reset."
        )

        self._modelcheck_prev_x = None
        self._modelcheck_prev_u_mpc = None
        self._modelcheck_prev_t = None

        self._diag_state_time = None

        # v3: clear recovery-mode state
        self._recovery_active = False
        self._recovery_since = None
        self.recovery_count = 0


# ============================================================
# ROS Entry Point
# ============================================================

if __name__ == "__main__":

    controller = (
        PhysicsMPCController()
    )

    controller.spin()