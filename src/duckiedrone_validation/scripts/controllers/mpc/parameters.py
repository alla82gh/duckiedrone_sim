#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parameters.py

Physics MPC Parameters

This module centralizes every parameter required by the
Physics MPC controller.

All MPC modules must import parameters from this file.

Author: Abdallah GHOUL 2026
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MPCParameters:
    """
    Physics MPC parameters.
    """

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    Ts: float = 0.01          # 100 Hz
    Np: int = 20              # Prediction Horizon
    Nc: int = 20              # Control Horizon

    # --------------------------------------------------
    # System Dimensions
    # --------------------------------------------------

    nx: int = 12              # State dimension
    nu: int = 4               # Inputs

    # --------------------------------------------------
    # Prediction-model physical parameters
    # --------------------------------------------------
    #
    # These are the parameters assumed by the Physics MPC
    # prediction model.
    #
    # At runtime physics_mpc_controller.py will overwrite
    # them from /dd21.  Therefore:
    #
    #   nominal run  -> nominal controller model
    #   S5 mismatch  -> mismatched controller model
    #
    # Gazebo plant parameters are NOT changed here.
    # --------------------------------------------------

    mass: float = 0.635       # [kg]
    gravity: float = 9.81     # [m/s^2]

    Ixx: float = 0.0015       # [kg.m^2]
    Iyy: float = 0.0017       # [kg.m^2]
    Izz: float = 0.0030       # [kg.m^2]

    # --------------------------------------------------
    # Cost Function Weights
    # --------------------------------------------------

    Q: np.ndarray = None
    R: np.ndarray = None
    S: np.ndarray = None
    P: np.ndarray = None

    # --------------------------------------------------
    # Solver
    # --------------------------------------------------

    max_iterations: int = 400
    tolerance: float = 1e-5

    # --------------------------------------------------
    # Constraints
    # --------------------------------------------------

    phi_max: float = np.deg2rad(25.0)
    theta_max: float = np.deg2rad(25.0)

    z_min: float = 0.0
    z_max: float = 5.0

    thrust_min: float = 0.0
    thrust_max: float = 15.0

    torque_max: float = 2.0

    du_max: float = 2.0

    # Per-channel control increment bounds:
    #
    #     [delta_T, tau_phi, tau_theta, tau_psi]
    #
    # None -> broadcast scalar du_max to all
    # four channels (backward compatible).
    du_max_vec: np.ndarray = None

    # --------------------------------------------------
    # Prediction-model coordinate offsets
    # --------------------------------------------------
    #
    # Coordinate convention:
    #
    #     x_absolute = x_model + state_offset
    #
    #     u_nominal  = u_model + input_offset
    #
    # Physics MPC:
    #
    #     state_offset = 0
    #     input_offset = 0
    #
    # PEM-MPC:
    #
    #     state_offset = x_op
    #     input_offset = u_trim_nominal
    #
    # Defaults remain zero so the validated Physics MPC
    # behavior is unchanged.
    # --------------------------------------------------

    state_offset: np.ndarray = None
    input_offset: np.ndarray = None

    # --------------------------------------------------
    # Soft state constraints
    # --------------------------------------------------

    soft_state_constraints: bool = True

    # Quadratic penalty applied to state-constraint
    # slack variables.
    #
    # Soft states:
    #     z, phi, theta
    #
    # J_soft = J_mpc + slack_weight * ||epsilon||^2
    #
    slack_weight: float = 1.0e4

    # --------------------------------------------------
    # Rotor allocation / actuator feasibility
    # --------------------------------------------------

    k_f: float = 8.54858e-06
    k_m: float = 1.0e-07

    arm_dx: float = 0.0775
    arm_dy: float = 0.1075

    max_rotor_velocity: float = 1000.0

    def __post_init__(self):

        if self.Q is None:

            self.Q = np.diag([10,10,20,
                              3,3,5,
                              10,10,10,
                              1,1,1])

        if self.R is None:

            self.R = np.diag([0.2,
                              0.05,
                              0.05,
                              0.05])

        if self.S is None:

            self.S = np.diag([0.02,
                              0.01,
                              0.01,
                              0.01])

        if self.P is None:

            self.P = self.Q.copy()

        # --------------------------------------------------
        # Per-channel control increment bounds
        # --------------------------------------------------

        if self.du_max_vec is None:

            self.du_max_vec = np.full(
                self.nu,
                self.du_max,
                dtype=float
            )

        self.du_max_vec = np.asarray(
            self.du_max_vec,
            dtype=float
        ).reshape(self.nu)

        if np.any(self.du_max_vec <= 0.0):

            raise ValueError(
                "du_max_vec entries must be positive."
            )

        # --------------------------------------------------
        # Prediction-model coordinate offsets
        # --------------------------------------------------

        if self.state_offset is None:

            self.state_offset = np.zeros(
                self.nx,
                dtype=float
            )

        self.state_offset = np.asarray(
            self.state_offset,
            dtype=float
        ).reshape(self.nx)

        if not np.all(
            np.isfinite(
                self.state_offset
            )
        ):
            raise ValueError(
                "state_offset contains NaN or Inf."
            )

        if self.input_offset is None:

            self.input_offset = np.zeros(
                self.nu,
                dtype=float
            )

        self.input_offset = np.asarray(
            self.input_offset,
            dtype=float
        ).reshape(self.nu)

        if not np.all(
            np.isfinite(
                self.input_offset
            )
        ):
            raise ValueError(
                "input_offset contains NaN or Inf."
            )