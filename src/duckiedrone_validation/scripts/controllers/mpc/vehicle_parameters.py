
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vehicle_parameters.py

Duckiedrone DD21 physical parameters.

This module centralizes all physical parameters used by the
Physics MPC controller.

Author: Abdallah Ghoul 2026
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleParameters:
    """
    Physical parameters of the Duckiedrone DD21.
    """

    # --------------------------------------------------
    # Physical properties
    # --------------------------------------------------

    gravity: float = 9.81          # [m/s²]

    mass: float = 0.635            # [kg]

    Ixx: float = 0.0015            # [kg·m²]
    Iyy: float = 0.0017            # [kg·m²]
    Izz: float = 0.0030            # [kg·m²]

    arm_length: float = 0.1075     # [m]

    # --------------------------------------------------
    # Propeller model
    # --------------------------------------------------

    kf: float = 8.54858e-06        # [N/(rad/s)^2]
    km: float = 1.0e-07            # [N·m/(rad/s)^2]

    max_rotor_velocity: float = 1000.0   # [rad/s]