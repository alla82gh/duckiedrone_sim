#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pid_controller.py — cascade PID baseline (thesis Chapter 3, Section 3.4).

Outer loop: position error -> desired accelerations -> attitude refs + thrust.
Inner loop: attitude error -> torques.  Anti-windup clamps on all integrators.
Gains from config/pid_gains.yaml (starting points -> thesis Table 4.2).

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

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class PID(object):
    def __init__(self, kp, ki, kd, i_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_max = i_max
        self.i = 0.0
        self.e_prev = 0.0
        self.first = True

    def step(self, e, dt):
        self.i = np.clip(self.i + e * dt, -self.i_max, self.i_max)
        d = 0.0 if self.first else (e - self.e_prev) / dt
        self.first = False
        self.e_prev = e
        return self.kp * e + self.ki * self.i + self.kd * d


class CascadePID(ControllerBase):
    def __init__(self):
        super(CascadePID, self).__init__("pid_controller")
        g = rospy.get_param("/pid")
        imax = g["integral_max"]
        self.px = PID(**g["pos_xy"], i_max=imax)
        self.py = PID(**g["pos_xy"], i_max=imax)
        self.pz = PID(**g["pos_z"], i_max=imax)
        self.ppsi = PID(**g["yaw"], i_max=imax)
        self.pphi = PID(**g["att_phi"], i_max=imax)
        self.ptheta = PID(**g["att_theta"], i_max=imax)
        self.ppsi_in = PID(**g["att_psi"], i_max=imax)
        self.att_ref_max = g["attitude_ref_max"]
        self.thrust_ff = g["thrust_ff"]
        self.dt = 1.0 / self.rate_hz

    def control_law(self, x, x_ref):
        # --- outer loop: position ---
        ax = self.px.step(x_ref[0] - x[0], self.dt)
        ay = self.py.step(x_ref[1] - x[1], self.dt)
        az = self.pz.step(x_ref[2] - x[2], self.dt)
        psi = x[5]
        # small-angle inversion (thesis Sec. 3.4): [ax;ay] = g*[theta;-phi] at psi=0,
        # rotated by current yaw:
        theta_d = (ax * np.cos(psi) + ay * np.sin(psi)) / self.g
        phi_d = (ax * np.sin(psi) - ay * np.cos(psi)) / self.g
        phi_d = np.clip(phi_d, -self.att_ref_max, self.att_ref_max)
        theta_d = np.clip(theta_d, -self.att_ref_max, self.att_ref_max)
        T = self.m * (self.g + az) if self.thrust_ff else self.m * self.g + az
        # --- inner loop: attitude ---
        tphi = self.pphi.step(wrap_angle(phi_d - x[3]), self.dt)
        ttheta = self.ptheta.step(wrap_angle(theta_d - x[4]), self.dt)
        tpsi = self.ppsi_in.step(wrap_angle(x_ref[5] - x[5]), self.dt)
        return np.array([T, tphi, ttheta, tpsi])


if __name__ == "__main__":
    CascadePID().spin()
