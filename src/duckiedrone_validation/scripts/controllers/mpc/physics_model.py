#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
physics_model.py

Physics-based prediction model
for the Duckiedrone.

This model will be used by:

- Physics MPC
- PEM-MPC (through the same interface)
- Hybrid PEM-GP MPC

Author: Abdallah GHOUL 2026
"""

import numpy as np

from .prediction_model import PredictionModel
from .linearization import Linearization


class PhysicsModel(PredictionModel):

    def __init__(self, parameters):

        super().__init__(parameters)

        # Prediction horizon
        self.N = self.params.Np

        # Linearization module
        self.linearization = Linearization(
            self.params
        )

        # Continuous and discrete models
        self.Ac, self.Bc, self.Ad, self.Bd = (
            self.linearization.get_discrete_model()
        )
    # --------------------------------------------------

    def linearize(self, x, u):
        """
        Compute discrete linear model.

        Returns
        -------
        A
        B
        """

        return self.Ad, self.Bd

    # --------------------------------------------------

    def predict(self, x, u):

        return self.Ad @ x + self.Bd @ u

    # --------------------------------------------------

    def hover_input(self):

        return np.zeros(self.nu)
    #--------------------------------------------------
    
    def build_phi(self):
        """
        Build the state prediction matrix Phi.
        """

        Phi = np.zeros((self.N * self.nx, self.nx))

        for i in range(self.N):
            Phi[
                i * self.nx:(i + 1) * self.nx,
                :
            ] = np.linalg.matrix_power(self.Ad, i + 1)

        return Phi
    
    def build_gamma(self):
        """
        Build the control prediction matrix Gamma.

        Returns
        -------
        ndarray
            Control prediction matrix.
        """

        Gamma = np.zeros((self.N * self.nx,
                        self.N * self.nu))

        for i in range(self.N):

            for j in range(i + 1):

                block = (
                    np.linalg.matrix_power(
                        self.Ad,
                        i - j
                    ) @ self.Bd
                )

                row = slice(
                    i * self.nx,
                    (i + 1) * self.nx
                )

                col = slice(
                    j * self.nu,
                    (j + 1) * self.nu
                )

                Gamma[row, col] = block

        return Gamma
    
    def build_prediction_model(self):
        """
        Returns
        -------
        Phi : ndarray
        Gamma : ndarray
        """
        return self.build_phi(), self.build_gamma()