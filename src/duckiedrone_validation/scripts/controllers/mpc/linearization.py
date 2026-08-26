"""
File:
    controllers/mpc/linearization.py

Description:
    Continuous and discrete linear model generation
    for the Physics-based MPC controller.

Author: Abdallah Ghoul  2026
"""
from __future__ import annotations

import numpy as np
from scipy.signal import cont2discrete

from .parameters import MPCParameters


class Linearization:
    """
    Generates the continuous and discrete linear
    state-space model around the hover equilibrium.
    """

    def __init__(self, parameters=None):

        # --------------------------------------------------
        # MPC / prediction-model parameters
        # --------------------------------------------------
        #
        # Runtime:
        #   PhysicsModel passes the MPCParameters instance
        #   populated from ROS /dd21.
        #
        # Offline/unit tests:
        #   If no object is provided, keep backward-compatible
        #   nominal defaults.
        # --------------------------------------------------

        if parameters is None:
            self.params = MPCParameters()
        else:
            self.params = parameters

        # --------------------------------------------------
        # System dimensions
        # --------------------------------------------------

        self.nx = self.params.nx
        self.nu = self.params.nu

        # --------------------------------------------------
        # Physical parameters assumed by prediction model
        # --------------------------------------------------

        self.g = float(self.params.gravity)
        self.m = float(self.params.mass)

        self.Ixx = float(self.params.Ixx)
        self.Iyy = float(self.params.Iyy)
        self.Izz = float(self.params.Izz)

        # --------------------------------------------------
        # Sampling time
        # --------------------------------------------------

        self.dt = float(self.params.Ts)

        # Validation
        if self.dt <= 0.0:
            raise ValueError("Sampling time must be positive.")

        if self.m <= 0.0:
            raise ValueError("Vehicle mass must be positive.")

        if min(self.Ixx, self.Iyy, self.Izz) <= 0.0:
            raise ValueError("Moments of inertia must be positive.")
        
    def build_Ac(self) -> np.ndarray:
        """
        Construct the continuous-time state matrix Ac
        around the hover equilibrium.

        Returns
        -------
        np.ndarray
            Continuous-time state matrix (12×12).
        """

        Ac = np.zeros((self.nx, self.nx), dtype=float)

        # --------------------------------------------------
        # Kinematics
        # --------------------------------------------------

        Ac[0, 6] = 1.0      # x_dot = vx
        Ac[1, 7] = 1.0      # y_dot = vy
        Ac[2, 8] = 1.0      # z_dot = vz

        Ac[3, 9]  = 1.0     # phi_dot   = p
        Ac[4, 10] = 1.0     # theta_dot = q
        Ac[5, 11] = 1.0     # psi_dot   = r

        # --------------------------------------------------
        # Translational Dynamics
        # --------------------------------------------------

        Ac[6, 4] = self.g      # vx_dot = g * theta
        Ac[7, 3] = -self.g     # vy_dot = -g * phi

        return Ac
    
    def build_Bc(self) -> np.ndarray:
        """
        Construct the continuous-time input matrix Bc
        around the hover equilibrium.

        Returns
        -------
        np.ndarray
            Continuous-time input matrix (12×4).
        """

        Bc = np.zeros((self.nx, self.nu), dtype=float)

        # --------------------------------------------------
        # Translational Dynamics
        # --------------------------------------------------

        Bc[8, 0] = 1.0 / self.m      # vz_dot = T / m

        # --------------------------------------------------
        # Rotational Dynamics
        # --------------------------------------------------

        Bc[9, 1] = 1.0 / self.Ixx    # p_dot = tau_phi / Ixx
        Bc[10, 2] = 1.0 / self.Iyy   # q_dot = tau_theta / Iyy
        Bc[11, 3] = 1.0 / self.Izz   # r_dot = tau_psi / Izz

        return Bc
    
    def discretize(
        self,
        Ac: np.ndarray,
        Bc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Discretize the continuous-time state-space model
        using Zero-Order Hold (ZOH).

        Parameters
        ----------
        Ac : np.ndarray
            Continuous-time state matrix.

        Bc : np.ndarray
            Continuous-time input matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Discrete-time state and input matrices (Ad, Bd).
        """

        # Output matrix
        C = np.zeros((self.nx, self.nx), dtype=float)

        # Feedthrough matrix
        D = np.zeros((self.nx, self.nu), dtype=float)

        Ad, Bd, _, _, _ = cont2discrete(
            (Ac, Bc, C, D),
            self.dt,
            method="zoh"
        )

        return Ad, Bd
    
    def get_discrete_model(
        self
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the continuous-time and discrete-time
        linear state-space model.

        Returns
        -------
        tuple
            (Ac, Bc, Ad, Bd)
        """

        # Continuous-time model
        Ac = self.build_Ac()
        Bc = self.build_Bc()

        # Discrete-time model
        Ad, Bd = self.discretize(Ac, Bc)

        return Ac, Bc, Ad, Bd