#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prediction_model.py

Abstract prediction model interface.

Every prediction model used by the MPC must inherit from this class.

Examples
--------
PhysicsModel
PEMModel
HybridModel

Author: Abdallah GHOUL 2026
"""

from abc import ABC, abstractmethod
import numpy as np


class PredictionModel(ABC):
    """
    Base interface for all prediction models.
    """

    def __init__(self, parameters):

        self.params = parameters

        self.nx = parameters.nx
        self.nu = parameters.nu

    @abstractmethod
    def linearize(self, x, u):
        """
        Compute the discrete linear model

            x(k+1)=Ax+Bu

        Returns
        -------
        A : ndarray
        B : ndarray
        """

        pass

    @abstractmethod
    def predict(self, x, u):
        """
        Predict one sampling step.

        Parameters
        ----------
        x : current state

        u : current input

        Returns
        -------
        next state
        """

        pass

    @abstractmethod
    def hover_input(self):
        """
        Return hover control input.

        Returns
        -------
        ndarray(4,)
        """

        pass