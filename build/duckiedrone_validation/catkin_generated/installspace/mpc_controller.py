#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpc_controller.py — conventional MPC (thesis Chapter 3, Section 3.5).
Constant reference held over the horizon (scenario_runner provides x_ref).

AUTHOR : Abdallah GHOUL  2026
"""
import numpy as np
import rospy
import os, sys
import rospkg
_PKG = rospkg.RosPack().get_path("duckiedrone_validation")
for _d in ("controllers", "models", "scenarios"):
    _p = os.path.join(_PKG, "scripts", _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from controller_base import ControllerBase
from mpc_core import LinearMPC


class MPCController(ControllerBase):
    def __init__(self):
        super(MPCController, self).__init__("mpc_controller")
        cfg = rospy.get_param("/mpc")
        p = rospy.get_param("/dd21")
        limits = dict(T_min=self.T_min, T_max=self.T_max,
                      tau_max=self.tau_max, du_max=self.du_max,
                      phi_max=p["phi_max"], theta_max=p["theta_max"],
                      z_max=p["z_max"])
        self.Np = cfg["Np"]
        self.mpc = LinearMPC(self.model, cfg, limits)
        self.infeasible_count = 0

    def control_law(self, x, x_ref):
        x_ref_traj = np.tile(x_ref, (self.Np + 1, 1))
        u, ok = self.mpc.solve(x, x_ref_traj)
        if not ok:
            self.infeasible_count += 1
            rospy.logwarn_throttle(1.0, "MPC infeasible (count=%d) — holding u_prev",
                                   self.infeasible_count)
        return u


if __name__ == "__main__":
    MPCController().spin()
