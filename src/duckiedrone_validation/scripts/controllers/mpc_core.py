#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpc_core.py — linear MPC QP, shared by the conventional MPC and the VS-TMPC.

Formulation (thesis Chapter 3, Section 3.5):
    min  sum_{i=0}^{Np-1} ||x_i - x_ref||_Q^2 + ||u_dev_i||_R^2
         + sum ||du_i||_S^2 + ||x_Np - x_ref||_{P}^2
    s.t. x_{i+1} = A x_i + B (u_i - u_hover)      [deviation form]
         |phi_i|,|theta_i| <= att_lim_i           [state box constraints]
         T_min <= T_i <= T_max, |tau_i| <= tau_max
         |u_i - u_{i-1}| <= du_max
Solved with CVXPY + OSQP (rebuilt each cycle; warm start via OSQP settings).
The VS-TMPC reuses the same builder with per-step tightened bounds.
"""
import numpy as np
import cvxpy as cp


class LinearMPC(object):
    def __init__(self, model, cfg, limits):
        self.model = model
        self.Np = cfg["Np"]; self.Nc = cfg["Nc"]; self.Ts = cfg["Ts"]
        self.Q = np.diag(cfg["Q"]); self.R = np.diag(cfg["R"])
        self.S = np.diag(cfg["S"]); self.P = cfg["P_scale"] * self.Q
        self.solver_opts = cfg.get("osqp", {})
        self.u_hover = model.hover_input()
        self.limits = limits          # dict: T_min,T_max,tau_max,du_max,phi_max,theta_max,z_max
        self.u_prev = self.u_hover.copy()

    def _state_bounds(self, i, r_i):
        """Per-step attitude/height limits, tightened by tube radius r_i."""
        att = max(self.limits["phi_max"] - r_i, 0.05)
        zmax = max(self.limits["z_max"] - r_i, 0.5)
        return att, zmax

    def solve(self, x0, x_ref_traj, tubes=None):
        """x_ref_traj: (Np+1, 12) reference; tubes: None or (Np+1,) radii."""
        Np, Nc = self.Np, self.Nc
        A, B = self.model.A, self.model.B
        uh = self.u_hover
        if tubes is None:
            tubes = np.zeros(Np + 1)

        X = cp.Variable((12, Np + 1))
        U = cp.Variable((4, Np))
        cost = 0
        cons = [X[:, 0] == x0]
        for i in range(Np):
            u_dev = U[:, i] - uh
            cons.append(X[:, i + 1] == A @ X[:, i] + B @ u_dev)
            e = X[:, i] - x_ref_traj[i]
            cost += cp.quad_form(e, self.Q) + cp.quad_form(u_dev, self.R)
            # input blocking beyond control horizon
            if i >= Nc:
                cons.append(U[:, i] == U[:, Nc - 1])
            # input-rate
            u_prev = self.u_prev if i == 0 else U[:, i - 1]
            du = U[:, i] - u_prev
            cost += cp.quad_form(du, self.S)
            # box constraints
            att, zmax = self._state_bounds(i, tubes[i])
            cons.append(X[3, i] >= -att); cons.append(X[3, i] <= att)
            cons.append(X[4, i] >= -att); cons.append(X[4, i] <= att)
            cons.append(X[2, i] >= 0.0);  cons.append(X[2, i] <= zmax)
            cons.append(U[0, i] >= self.limits["T_min"])
            cons.append(U[0, i] <= self.limits["T_max"])
            cons.append(U[1:4, i] >= -self.limits["tau_max"])
            cons.append(U[1:4, i] <= self.limits["tau_max"])
            cons.append(du >= -self.limits["du_max"])
            cons.append(du <= self.limits["du_max"])
        eT = X[:, Np] - x_ref_traj[Np]
        cost += cp.quad_form(eT, self.P)

        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=self.solver_opts.get("warm_start", True),
                   max_iter=self.solver_opts.get("max_iter", 4000),
                   eps_abs=self.solver_opts.get("eps_abs", 1e-4),
                   eps_rel=self.solver_opts.get("eps_rel", 1e-4),
                   polish=self.solver_opts.get("polish", True))
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # graceful fallback: keep previous input (logged as infeasible)
            return self.u_prev.copy(), False
        u0 = np.array(U[:, 0].value).flatten()
        self.u_prev = u0
        return u0, True

    def reference_preview(self, x_ref, i):
        return x_ref
