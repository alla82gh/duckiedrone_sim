#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vstmpc_controller.py — Variance-Scheduled Tube MPC (thesis Chapter 3,
Section 3.6, Algorithm 1).

Per control cycle:
  1. read state estimate, reference
  2. propagate covariance  Sigma_{i+1} = A Sigma_i A^T + Sigma_GP,i  (3.42)
  3. confidence metric     gamma_i = sqrt(lambda_max(Sigma_i))       (3.44)
  4. tube radius           r_i = clip(r_min + alpha*gamma_i,
                                      r_min, r_max)                  (3.46)
  5. solve nominal MPC with constraints tightened by r_i
  6. apply first input; publish gamma, r for the thesis traces (Fig. 4.4)

  AUTHOR : Abdallah GHOUL  2026
"""
import numpy as np
import rospy
from controller_base import ControllerBase
from mpc_core import LinearMPC

import os, sys
import rospkg
_PKG = rospkg.RosPack().get_path("duckiedrone_validation")
for _d in ("controllers", "models", "scenarios"):
    _p = os.path.join(_PKG, "scripts", _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gp_uncertainty import GPUncertainty, propagate_covariance, confidence_metric


class VSTMPCController(ControllerBase):
    def __init__(self):
        super(VSTMPCController, self).__init__("vstmpc_controller")
        mpc_cfg = rospy.get_param("/mpc")
        v = rospy.get_param("/vstmpc")
        p = rospy.get_param("/dd21")
        limits = dict(T_min=self.T_min, T_max=self.T_max,
                      tau_max=self.tau_max, du_max=self.du_max,
                      phi_max=p["phi_max"], theta_max=p["theta_max"],
                      z_max=p["z_max"])
        self.Np = mpc_cfg["Np"]
        self.mpc = LinearMPC(self.model, mpc_cfg, limits)
        self.r_min, self.r_max, self.alpha = v["r_min"], v["r_max"], v["alpha"]
        gp = v["gp"]
        self.gp_unc = GPUncertainty(mode=gp["mode"],
                                    sigma2_base=gp["sigma2_base"],
                                    sigma2_dyn=gp["sigma2_dyn"],
                                    pickle_path=gp.get("pickle_path", ""))
        self.infeasible_count = 0

    def control_law(self, x, x_ref):
        # --- uncertainty propagation & tube scheduling (steps 2-4) ---
        Sigmas = propagate_covariance(self.model, self.gp_unc, x,
                                      self.mpc.u_hover, self.Np)
        gammas = np.array([confidence_metric(S) for S in Sigmas])
        tubes = np.clip(self.r_min + self.alpha * gammas,
                        self.r_min, self.r_max)
        self.gamma = float(gammas[1])          # one-step-ahead confidence
        self.r = float(tubes[1])               # applied tube radius
        # --- tightened nominal MPC (step 5) ---
        x_ref_traj = np.tile(x_ref, (self.Np + 1, 1))
        u, ok = self.mpc.solve(x, x_ref_traj, tubes=tubes)
        if not ok:
            self.infeasible_count += 1
            rospy.logwarn_throttle(1.0,
                "VS-TMPC infeasible (count=%d) — holding u_prev",
                self.infeasible_count)
        return u


if __name__ == "__main__":
    VSTMPCController().spin()
