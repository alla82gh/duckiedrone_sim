#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_runner.py — runs scenarios S1-S5 (thesis Section 4.2.4) and logs
everything needed for Tables 4.2-4.4 and Figures 4.2-4.4.

Phases: takeoff -> scenario -> land -> disarm (writes CSVs).
Publishes: /validation/reference_state (std_msgs/Float32MultiArray, 12-dim).
Subscribes: odometry topic (state) and /validation/control_log (u, gamma, r, tc).

Scenarios:
  S1 hover:               hold (0,0,1.5) for hover_duration
  S2 attitude steps:      roll +10deg (4 s), pitch +10deg (4 s),
                          yaw +45deg (5 s ramp up, hold, ramp down
                          during landing)
  S3 trajectory:          circle radius R, altitude 1.5, period T_circle
  S4 disturbance:         hover + lateral force pulses via ApplyBodyWrench
  S5 parametric mismatch: identical to S3; the MISMATCH is applied at launch
                          time (run_scenario.launch mismatch:=true loads
                          config/dd21_params_mismatched.yaml into /dd21)

Outputs (results/):
  {controller}_{scenario}_run{id}_series.csv   full time series
  summary.csv                                  one appended row per run
"""
import os, csv, time
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest
from geometry_msgs.msg import Wrench, Vector3

import os.path as osp
import sys
import rospkg
_PKG = rospkg.RosPack().get_path("duckiedrone_validation")
for _d in ("controllers", "models", "scenarios"):
    _p = osp.join(_PKG, "scripts", _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
import metrics
from controller_base import quat_to_euler


def ramp(t, T):
    return np.clip(t / T, 0.0, 1.0)


class ScenarioRunner(object):
    def __init__(self):
        rospy.init_node("scenario_runner")
        self.ctrl = rospy.get_param("~controller", "pid")
        self.scn = rospy.get_param("~scenario", "S1").upper()
        self.run_id = rospy.get_param("~run_id", 1)
        gz = rospy.get_param("/gazebo")
        tp = rospy.get_param("/topics")
        self.model_name = gz["model_name"]
        self.odom_topic = tp["odom"]
        self.rate_hz = rospy.get_param("/dd21/control_rate", 100.0)
        # scenario timing [s]
        self.T_takeoff = 4.0
        self.T_land = 4.0
        self.hover_z = 1.5
        # --------------------------------------------------
        # Scenario timing [s]
        # --------------------------------------------------
        self.durations = {
            "S1": 30.0,
            "S2": 2.0,     # overwritten below for S2 yaw
            "S3": 40.0,
            "S4": 30.0,
            "S5": 40.0,
        }

        self.circle_R = 2
        self.circle_T = 50.0

        # --------------------------------------------------
        # S2 is split into three independent experiments:
        #
        #   roll  : +10 deg roll for 2 s
        #   pitch : +10 deg pitch for 2 s
        #   yaw   : +45 deg yaw, 5 s ramp + 3 s hold
        #
        # Each experiment starts from a fresh hover/run.
        #
        # The 2 s roll/pitch window is intentionally short.
        # A sustained 10 deg tilt produces about 1.7 m/s^2
        # lateral acceleration; reducing the hold from 4 s
        # to 2 s limits the theoretical displacement from
        # about 13.8 m to about 3.4 m while remaining long
        # enough for attitude-response metrics.
        # --------------------------------------------------
        self.s2_axis = str(
            rospy.get_param("~s2_axis", "roll")
        ).strip().lower()

        if self.s2_axis not in ("roll", "pitch", "yaw"):
            raise ValueError(
                "~s2_axis must be one of: roll, pitch, yaw"
            )

        self.s2_attitude_T = 2.0
        self.yaw_ramp_T = 5.0
        self.s2_yaw_hold_T = 3.0

        if self.scn == "S2":
            if self.s2_axis in ("roll", "pitch"):
                self.durations["S2"] = self.s2_attitude_T
            else:
                self.durations["S2"] = (
                    self.yaw_ramp_T + self.s2_yaw_hold_T
                )

            rospy.loginfo(
                "runner: S2 subtest = %s, duration = %.1f s",
                self.s2_axis,
                self.durations["S2"],
            )
        self.gust_defs = [  # (time_in_scenario, Fx, Fy, duration)
            (8.0, 2.0, 0.0, 0.4),
            (16.0, 0.0, 2.0, 0.4),
            (24.0, -2.0, -2.0, 0.4),
        ]                                               # [CONFIRM: gust magnitudes]
        self.x = np.zeros(12); self.have = False
        self.ulog = np.zeros(4); self.gamma = 0.0; self.r = 0.0; self.tc = 0.0
        self.ref_pub = rospy.Publisher("/validation/reference_state",
                                       Float32MultiArray, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/validation/control_log", Float32MultiArray,
                         self._log_cb, queue_size=1)
        self.outdir = osp.join(_PKG, "results")
        os.makedirs(self.outdir, exist_ok=True)

    def _odom_cb(self, m):
        phi, th, psi = quat_to_euler(m.pose.pose.orientation)
        self.x = np.array([m.pose.pose.position.x, m.pose.pose.position.y,
                           m.pose.pose.position.z, phi, th, psi,
                           m.twist.twist.linear.x, m.twist.twist.linear.y,
                           m.twist.twist.linear.z, m.twist.twist.angular.x,
                           m.twist.twist.angular.y, m.twist.twist.angular.z])
        self.have = True

    def _log_cb(self, m):
        d = m.data
        self.ulog = np.array(d[0:4]); self.gamma = d[4]; self.r = d[5]
        self.tc = d[6]

    # ---------- reference profiles ----------
    def reference(self, ts):
        """ts: time inside the scenario phase. Returns 12-dim x_ref."""
        xr = np.zeros(12)
        if self.scn in ("S1", "S4"):
            xr[2] = self.hover_z
        elif self.scn == "S2":
            xr[2] = self.hover_z

            if self.s2_axis == "roll":
                # Independent S2-Roll experiment.
                xr[3] = np.radians(10.0)

            elif self.s2_axis == "pitch":
                # Independent S2-Pitch experiment.
                xr[4] = np.radians(10.0)

            elif self.s2_axis == "yaw":
                # Independent S2-Yaw experiment:
                # 5 s ramp to +45 deg, then 3 s hold.
                #
                # Kinematic consistency:
                #
                #     psi_dot = r
                #
                # Therefore during the ramp:
                #
                #     r_ref = 45 deg / 5 s
                #           = 9 deg/s
                #           = 0.1570796 rad/s
                #
                # During the hold:
                #
                #     r_ref = 0
                # ------------------------------------------------

                psi_ref = np.radians(45.0)

                if ts < self.yaw_ramp_T:

                    xr[5] = psi_ref * ramp(
                        ts,
                        self.yaw_ramp_T
                    )

                    xr[11] = (
                        psi_ref
                        / self.yaw_ramp_T
                    )

                else:

                    xr[5] = psi_ref
                    xr[11] = 0.0
        elif self.scn in ("S3", "S5"):
            w = 2.0 * np.pi / self.circle_T
            xr[0] = self.circle_R * np.cos(w * ts)
            xr[1] = self.circle_R * np.sin(w * ts)
            xr[2] = self.hover_z
            xr[6] = -self.circle_R * w * np.sin(w * ts)
            xr[7] = self.circle_R * w * np.cos(w * ts)
        elif self.scn == "S6":
            xr[2] = self.hover_z

            if ts < 5.0:
                # Hover at P1
                xr[0] = 0.0
                xr[1] = 0.0

            elif ts < 10.0:
                # Move to P2
                xr[0] = 1.0
                xr[1] = 0.0

            elif ts < 15.0:
                # Move to P3
                xr[0] = 1.0
                xr[1] = 1.0

            elif ts < 20.0:
                # Move to P4
                xr[0] = 0.0
                xr[1] = 1.0

            else:
                # Return to P1
                xr[0] = 0.0
                xr[1] = 0.0
        return xr

    def apply_gusts(self, ts, applied):
        if self.scn != "S4":
            return
        try:
            rospy.wait_for_service("/gazebo/apply_body_wrench", timeout=1.0)
            srv = rospy.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)
        except Exception:
            rospy.logwarn_throttle(5.0, "apply_body_wrench unavailable")
            return
        for k, (t0, fx, fy, dur) in enumerate(self.gust_defs):
            if k in applied or not (t0 <= ts < t0 + 0.05):
                continue
            req = ApplyBodyWrenchRequest()
            req.body_name = self.model_name + "::base_link"  # [ADAPT link name]
            req.wrench = Wrench(force=Vector3(fx, fy, 0.0),
                                torque=Vector3(0.0, 0.0, 0.0))
            req.duration = rospy.Duration(dur)
            try:
                srv(req)
                rospy.loginfo("gust %d applied: Fx=%.1f Fy=%.1f", k, fx, fy)
            except Exception as e:
                rospy.logwarn("gust failed: %s", e)
            applied.add(k)

    # ---------- main run ----------
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not self.have and not rospy.is_shutdown():
            rate.sleep()

        # ----------------------------------------------------------
        # Wait until the active controller has completed
        # initialization and has produced its first control command.
        #
        # This prevents the takeoff reference clock from starting
        # while Physics MPC is still building/initializing its QP.
        # ----------------------------------------------------------
        rospy.loginfo(
            "runner: waiting for controller readiness..."
        )

        try:
            rospy.wait_for_message(
                "/validation/control_log",
                Float32MultiArray,
                timeout=15.0
            )

            rospy.loginfo(
                "runner: controller ready — starting scenario clock"
            )

        except rospy.ROSException:
            raise RuntimeError(
                "Controller did not become ready within 15 s."
            )

        rec = []
        phases = []

        # Start scenario timing ONLY after the controller is ready.
        t0 = time.time()
        Tscn = self.durations.get(self.scn, 30.0)
        gust_applied = set()
        rospy.loginfo("runner: %s / %s / run %d — takeoff", self.ctrl,
                      self.scn, self.run_id)
        while not rospy.is_shutdown():
            t = time.time() - t0
            if t < self.T_takeoff:
                phase = "takeoff"
                xr = np.zeros(12); xr[2] = self.hover_z * ramp(t, self.T_takeoff)
            elif t < self.T_takeoff + Tscn:
                phase = "scenario"
                ts = t - self.T_takeoff
                xr = self.reference(ts)
                self.apply_gusts(ts, gust_applied)
            elif t < self.T_takeoff + Tscn + self.T_land:
                phase = "land"
                tl = t - self.T_takeoff - Tscn
                xr = np.zeros(12)
                xr[2] = self.hover_z * (1.0 - ramp(tl, self.T_land))
                if self.scn == "S2" and self.s2_axis == "yaw":
                    # Return yaw smoothly to zero during landing.
                    psi_end = np.radians(45.0)
                    xr[5] = psi_end * (
                        1.0 - ramp(tl, self.T_land)
                    )
            else:
                break
            msg = Float32MultiArray(); msg.data = xr.tolist()
            self.ref_pub.publish(msg)
            rec.append([t] + self.x.tolist() + xr.tolist()
                       + self.ulog.tolist() + [self.gamma, self.r, self.tc])
            phases.append(phase)
            rate.sleep()
        self._save(np.array(rec), np.array(phases))

    def _save(self, rec, phases):
        head = (["t", "x", "y", "z", "phi", "theta", "psi", "dx", "dy", "dz",
                 "p", "q", "r", "xref", "yref", "zref", "phiref", "thetaref",
                 "psiref", "dxref", "dyref", "dzref", "pref", "qref", "rref",
                 "u_T", "u_tphi", "u_ttheta", "u_tpsi", "gamma", "r",
                 "tc_ms"])
        scenario_label = self.scn
        if self.scn == "S2":
            scenario_label = "S2_%s" % self.s2_axis

        f_series = osp.join(
            self.outdir,
            "%s_%s_run%02d_series.csv"
            % (self.ctrl, scenario_label, self.run_id)
        )
        with open(f_series, "w", newline="") as f:
            w = csv.writer(f); w.writerow(head + ["phase"])
            for row, ph in zip(rec.tolist(), phases.tolist()):
                w.writerow(row + [ph])
        t = rec[:, 0]; xs = rec[:, 1:13]; xr = rec[:, 13:25]
        us = rec[:, 25:29]; gm = rec[:, 29]; rr = rec[:, 30]; cm = rec[:, 31]
        row = metrics.summarize(t, xs, xr, us, cm, phases, gamma=gm, r=rr)
        row.update({
            "controller": self.ctrl,
            "scenario": scenario_label,
            "run_id": self.run_id,
            "duration_s": float(t[-1]),
        })
        f_sum = osp.join(self.outdir, "summary.csv")
        write_head = not osp.exists(f_sum)
        with open(f_sum, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_head:
                w.writeheader()
            w.writerow(row)
        rospy.loginfo("saved %s and appended summary.csv", f_series)


if __name__ == "__main__":
    ScenarioRunner().run()