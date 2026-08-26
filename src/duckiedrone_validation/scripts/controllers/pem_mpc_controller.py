#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_mpc_controller.py

PEM-based MPC controller integration
for the Duckiedrone DD21 validation framework.

This controller reuses the validated Physics MPC infrastructure:

    CostFunction
    Constraints
    Optimizer
    soft-QP
    rotor feasibility
    Delta-U handling
    recovery
    fail-safe
    diagnostics
    ROS ControllerBase

The only prediction-model change is:

    PhysicsModel
        ->
    PEMModel

The PEM model operates in fixed training-centered coordinates:

    x_model =
        x_absolute - x_op

    u_model =
        u_nominal - u_trim_nominal

where:

    u_nominal =
        [
            T - T_hover,
            tau_phi,
            tau_theta,
            tau_psi
        ]

Author: Abdallah GHOUL 2026
"""

import numpy as np
import rospy


# ============================================================
# Imports compatible with:
#
#   python -m controllers....
#
# and ROS direct script execution.
# ============================================================

try:

    from .physics_mpc_controller import (
        PhysicsMPCController
    )

    from .mpc.parameters import (
        MPCParameters
    )

    from .mpc.pem_model import (
        PEMModel
    )

except ImportError:

    from physics_mpc_controller import (
        PhysicsMPCController
    )

    from mpc.parameters import (
        MPCParameters
    )

    from mpc.pem_model import (
        PEMModel
    )


# ============================================================
# PEM MPC Controller
# ============================================================

class PEMMPCController(
    PhysicsMPCController
):
    """
    PEM-MPC controller for the Duckiedrone DD21.

    All common MPC and safety logic is inherited from
    PhysicsMPCController.

    Only:

        prediction model
        state operating-point offset
        input trim offset

    differ from Physics MPC.
    """

    def __init__(self):

        # ----------------------------------------------------
        # Load the frozen PEM deployment coordinates once
        # before constructing the common MPC infrastructure.
        #
        # MPCParameters() defaults are sufficient here because
        # PEMModel only needs the model dimensions / Ts to load
        # and validate its frozen artifact.
        # ----------------------------------------------------

        bootstrap_parameters = (
            MPCParameters()
        )

        bootstrap_model = (
            PEMModel(
                bootstrap_parameters
            )
        )

        state_offset = (
            bootstrap_model.x_op.copy()
        )

        input_offset = (
            bootstrap_model.u_trim_nominal.copy()
        )

        # ----------------------------------------------------
        # Construct the common MPC controller.
        #
        # The parent class will:
        #
        #   * load runtime /mpc and /dd21 parameters
        #   * create PEMModel with runtime parameters
        #   * build Phi / Gamma
        #   * create both optimizers
        #   * apply offset-aware constraints
        #   * initialize model-coordinate input memory
        # ----------------------------------------------------

        super(
            PEMMPCController,
            self
        ).__init__(
            controller_name="pem_mpc_controller",
            prediction_model_class=PEMModel,
            state_offset=state_offset,
            input_offset=input_offset
        )

        # ----------------------------------------------------
        # Integration consistency checks
        # ----------------------------------------------------

        if not isinstance(
            self.model,
            PEMModel
        ):

            raise RuntimeError(
                "PEM-MPC prediction model was not "
                "constructed as PEMModel."
            )

        if not np.allclose(
            self.parameters.state_offset,
            self.model.x_op,
            atol=1.0e-12,
            rtol=0.0
        ):

            raise RuntimeError(
                "PEM-MPC state_offset does not match "
                "the frozen PEM deployment operating point."
            )

        if not np.allclose(
            self.parameters.input_offset,
            self.model.u_trim_nominal,
            atol=1.0e-12,
            rtol=0.0
        ):

            raise RuntimeError(
                "PEM-MPC input_offset does not match "
                "the frozen PEM deployment trim."
            )

        # ----------------------------------------------------
        # Physical wrench represented by ZERO PEM input.
        #
        # This is NOT exactly nominal mg hover because the
        # identified model has a small training-only trim:
        #
        #     u_model = 0
        #
        # corresponds to:
        #
        #     [T_hover, 0, 0, 0]
        #       + u_trim_nominal
        # ----------------------------------------------------

        self.pem_zero_input_physical = (
            self._model_to_physical_input(
                np.zeros(
                    self.parameters.nu,
                    dtype=float
                )
            )
        )

        # ----------------------------------------------------
        # Startup information
        # ----------------------------------------------------

        rospy.loginfo(
            "=== PEM MPC Controller initialized ==="
        )

        rospy.loginfo(
            "PEM spectral radius: %.9f",
            float(
                self.model.spectral_radius
            )
        )

        rospy.loginfo(
            "PEM state offset norm: %.9f",
            float(
                np.linalg.norm(
                    self.parameters.state_offset
                )
            )
        )

        rospy.loginfo(
            "PEM input trim: "
            "[%+.9f %+.9f %+.9f %+.9f]",
            self.parameters.input_offset[0],
            self.parameters.input_offset[1],
            self.parameters.input_offset[2],
            self.parameters.input_offset[3]
        )

        rospy.loginfo(
            "PEM zero-model-input physical wrench: "
            "[%.9f %+.9f %+.9f %+.9f]",
            self.pem_zero_input_physical[0],
            self.pem_zero_input_physical[1],
            self.pem_zero_input_physical[2],
            self.pem_zero_input_physical[3]
        )


# ============================================================
# ROS Entry Point
# ============================================================

if __name__ == "__main__":

    controller = (
        PEMMPCController()
    )

    controller.spin()