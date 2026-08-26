#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dynamic_model.py

Continuous linearized dynamics around hover equilibrium.

This class contains only the physical parameters and the
continuous-time dynamic equations.

No linearization.
No discretization.
No optimization.

Author: Abdallah GHOUL 2026
"""

import numpy as np
import rospy


class DynamicModel:

    def __init__(self, vehicle):

        # --------------------------------------------------
        # Physical Parameters
        # --------------------------------------------------

        self.m = vehicle.mass

        self.g = 9.81

        self.Ixx = vehicle.Ixx
        self.Iyy = vehicle.Iyy
        self.Izz = vehicle.Izz

        self.l = vehicle.arm_length

        self.kf = vehicle.k_f
        self.km = vehicle.k_m

    # --------------------------------------------------

    def parameters(self):
        """
        Return vehicle parameters.
        """

        return {
            "m": self.m,
            "g": self.g,
            "Ixx": self.Ixx,
            "Iyy": self.Iyy,
            "Izz": self.Izz,
            "l": self.l,
            "kf": self.kf,
            "km": self.km,
        }
    
    # --------------------------------------------------

    def _kinematics(self, x_dot, x):
        """
        Position and attitude kinematics.
        """

        # Position
        x_dot[0] = x[6]      # x_dot = vx
        x_dot[1] = x[7]      # y_dot = vy
        x_dot[2] = x[8]      # z_dot = vz

        # Euler angles
        x_dot[3] = x[9]      # phi_dot = p
        x_dot[4] = x[10]     # theta_dot = q
        x_dot[5] = x[11]     # psi_dot = r

        return x_dot
    
    # --------------------------------------------------

    def _translational_dynamics(self, x_dot, x, u):
        """
        Translational dynamics around hover.
        """

        phi = x[3]
        theta = x[4]

        T = u[0]

        # Linearized translational dynamics

        x_dot[6] = self.g * theta

        x_dot[7] = -self.g * phi

        x_dot[8] = (T / self.m) - self.g

        return x_dot
    
    # --------------------------------------------------

    def _rotational_dynamics(self, x_dot, u):
        """
        Rotational dynamics around hover.
        """

        tau_phi = u[1]
        tau_theta = u[2]
        tau_psi = u[3]

        x_dot[9] = tau_phi / self.Ixx
        x_dot[10] = tau_theta / self.Iyy
        x_dot[11] = tau_psi / self.Izz

        return x_dot
    
    # --------------------------------------------------

    def dynamics(self, x, u):
        """
        Continuous-time nonlinear dynamics.

        Parameters
        ----------
        x : ndarray (12,)
            State vector.

        u : ndarray (4,)
            Control input.

        Returns
        -------
        x_dot : ndarray (12,)
        """

        # --------------------------------------------------
        # State vector
        # --------------------------------------------------

        px, py, pz = x[0:3]

        phi, theta, psi = x[3:6]

        vx, vy, vz = x[6:9]

        p, q, r = x[9:12]

        # --------------------------------------------------
        # Control input
        # --------------------------------------------------

        T = u[0]

        tau_phi = u[1]

        tau_theta = u[2]

        tau_yaw = u[3]

        # --------------------------------------------------
        # State derivative
        # --------------------------------------------------

        x_dot = np.zeros(12)

        x_dot = self._kinematics(x_dot, x)

        x_dot = self._translational_dynamics(x_dot, x, u)

        x_dot = self._rotational_dynamics(x_dot, u)

        return x_dot