#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quadrotor_model.py — nominal DD21 model used by the predictive controllers.

Implements:
  * LinearizedModel: hover-linearized 12-state model, exact ZOH discretization.
    State  x = [x y z phi theta psi dx dy dz p q r]^T
    Input  u = [T tphi ttheta tpsi]^T  (absolute thrust; mg feedforward inside)
    Matches the thesis notation (Chapter 3).
  * Allocation mixer: u -> 4 rotor velocities (rad/s), standard quad-X layout:
        T     = k_f (w1^2 + w2^2 + w3^2 + w4^2)
        tphi  = k_f l (w2^2 - w4^2)
        ttheta= k_f l (w3^2 - w1^2)
        tpsi  = k_m (w1^2 - w2^2 + w3^2 - w4^2)
    [ADAPT: rotor numbering/signs to your DD21 Gazebo model convention.]
    
AUTHOR: ABDALLAH GHOUL  2026
"""
import numpy as np
from scipy.linalg import expm


class LinearizedModel(object):
    def __init__(self, m, Ixx, Iyy, Izz, Ts, g=9.81):
        self.m, self.Ixx, self.Iyy, self.Izz = m, Ixx, Iyy, Izz
        self.Ts, self.g = Ts, g
        self.nx, self.nu = 12, 4
        self._build()
        self._discretize()

    def _build(self):
        m, Ixx, Iyy, Izz, g = self.m, self.Ixx, self.Iyy, self.Izz, self.g
        A = np.zeros((12, 12))
        # kinematics: pos <- vel ; att <- rates
        A[0, 6] = 1.0; A[1, 7] = 1.0; A[2, 8] = 1.0
        A[3, 9] = 1.0; A[4, 10] = 1.0; A[5, 11] = 1.0
        # small-angle translational dynamics about hover (psi0 = 0)
        A[6, 4] = g          # x_ddot  <- theta
        A[7, 3] = -g         # y_ddot  <- phi
        # rotational dynamics handled via B (1/I)
        B = np.zeros((12, 4))
        B[8, 0] = 1.0 / m    # z_ddot  <- T  (deviation from hover thrust)
        B[9, 1] = 1.0 / Ixx  # p_dot   <- tphi
        B[10, 2] = 1.0 / Iyy # q_dot   <- ttheta
        B[11, 3] = 1.0 / Izz # r_dot   <- tpsi
        self.Ac, self.Bc = A, B

    def _discretize(self):
        """Exact ZOH discretization via matrix exponential."""
        n = self.nx + self.nu
        M = np.zeros((n, n))
        M[:self.nx, :self.nx] = self.Ac
        M[:self.nx, self.nx:] = self.Bc
        Md = expm(M * self.Ts)
        self.A = Md[:self.nx, :self.nx]
        self.B = Md[:self.nx, self.nx:]

    def jacobian_at(self, x, u):
        """Local Jacobian of the hybrid prediction (nominal part).
        For the linearized model it is simply (A, B); kept as a method so the
        VS-TMPC covariance propagation mirrors thesis Eq. (3.42)."""
        return self.A, self.B

    def hover_input(self):
        return np.array([self.m * self.g, 0.0, 0.0, 0.0])


class AllocationMixer(object):
    """u=[T,tphi,ttheta,tpsi] -> rotor velocities [rad/s], rectangular quad:
       M1 rear-right, M2 front-right, M3 rear-left, M4 front-left.
       Spins (confirmed from hardware): M1,M4 CW ; M2,M3 CCW."""
    def __init__(self, k_f, k_m, dx, dy, w_max):
        self.k_f, self.k_m, self.dx, self.dy, self.w_max = k_f, k_m, dx, dy, w_max
        M = np.array([
            [ k_f,      k_f,      k_f,      k_f     ],
            [-k_f*dy,  -k_f*dy,   k_f*dy,   k_f*dy  ],
            [ k_f*dx,  -k_f*dx,   k_f*dx,  -k_f*dx  ],
            [ k_m,     -k_m,     -k_m,      k_m     ],
        ])
        self.Minv = np.linalg.inv(M)

    def to_rotors(self, u):
        w2 = self.Minv @ np.asarray(u, float).reshape(4)
        return np.sqrt(np.clip(w2, 0.0, self.w_max**2))
