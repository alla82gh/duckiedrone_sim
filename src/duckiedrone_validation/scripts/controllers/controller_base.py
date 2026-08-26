#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controller_base.py — shared ROS scaffolding for the three controllers.

Common behavior (per thesis Figure 4.1, center panel):
  * loads DD21 params + controller params from the parameter server (YAML)
  * subscribes to the odometry topic and to /validation/reference_state
    (std_msgs/Float32MultiArray, 12 values, published by scenario_runner)
  * runs the control law at dd21/control_rate
  * publishes rotor commands on topics/cmd_motors and a per-cycle log on
    /validation/control_log: [u0..u3, gamma, r, t_compute_ms]
  * optional arming through the /enable_motors service

  AUTHOR : Abdallah GHOUL  2026
"""
import threading
import time
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

from std_srvs.srv import SetBool

import os, sys
import rospkg
_PKG = rospkg.RosPack().get_path("duckiedrone_validation")
for _d in ("controllers", "models", "scenarios"):
    _p = os.path.join(_PKG, "scripts", _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from quadrotor_model import LinearizedModel, AllocationMixer  # type: ignore

def quat_to_euler(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    t0 = 2.0 * (w * x + y * z); t1 = 1.0 - 2.0 * (x * x + y * y)
    phi = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    theta = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y); t4 = 1.0 - 2.0 * (y * y + z * z)
    psi = np.arctan2(t3, t4)
    return phi, theta, psi


class ControllerBase(object):
    def __init__(self, node_name):
        rospy.init_node(node_name)
        p = rospy.get_param("/dd21")
        t = rospy.get_param("/topics")
        self.m = p["mass"]; self.Ixx = p["Ixx"]; self.Iyy = p["Iyy"]
        self.Izz = p["Izz"]; self.g = p["gravity"]
        self.T_min = p["T_min"]; self.T_max = p["T_max"]
        self.tau_max = p["tau_max"]; self.du_max = p["du_max"]
        self.rate_hz = p["control_rate"]
        self.model = LinearizedModel(self.m, self.Ixx, self.Iyy, self.Izz,
                                     1.0 / self.rate_hz, self.g)
        self.mixer = AllocationMixer(p["k_f"], p["k_m"], p.get("arm_dx", 0.0775),
                                     p.get("arm_dy", 0.1075), p["max_rotor_velocity"])
        self.x = np.zeros(12)
        self.x_ref = np.zeros(12)
        self.have_odom = False

        # ----------------------------------------------------
        # Fresh-odometry gating (Fix B, 2026-08-23)
        #
        # When True (enabled by Physics MPC), the control loop
        # solves only on a NEWLY arrived state snapshot and holds
        # the last applied wrench otherwise. PID keeps the default
        # False and is completely unaffected.
        # ----------------------------------------------------
        self.require_fresh_odom = False

        # Set by _odom_cb on every new odometry message; the
        # event-driven control path waits on it (MPC only).
        self._odom_event = threading.Event()

        # ----------------------------------------------------
        # Odometry/control timing diagnostics
        # ----------------------------------------------------

        self.odom_seq = 0
        self.odom_stamp = None
        self.odom_arrival_time = None

        self.last_control_odom_seq = -1
        # Odometry callback-rate diagnostic
        self._odom_diag_last_seq = 0
        self._odom_diag_last_time = rospy.get_time()
        self.u_prev = self.model.hover_input()
        self.cmd_pub = rospy.Publisher(t["cmd_motors"], Float32MultiArray,
                                       queue_size=1)
        self.log_pub = rospy.Publisher("/validation/control_log",
                                       Float32MultiArray, queue_size=10)
        rospy.Subscriber(t["odom"], Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/validation/reference_state", Float32MultiArray,
                         self._ref_cb, queue_size=1)
        self.gamma, self.r = 0.0, 0.0     # reported by VS-TMPC
        self.comp_ms = 0.0
        if t.get("use_enable_service", True):
            self._arm(t["enable_motors"])

    def _arm(self, srv_name):
        """Arm the propulsion chain via /enable_motors (mixer_node)."""
        try:
            rospy.wait_for_service(srv_name, timeout=5.0)
            arm = rospy.ServiceProxy(srv_name, SetBool)
            arm(True)
            rospy.loginfo("[%s] motors armed via %s", rospy.get_name(), srv_name)
        except Exception:
            rospy.logwarn("[%s] %s unavailable — assuming direct drive",
                          rospy.get_name(), srv_name)

    def _odom_cb(self, msg):
        """
        Convert odometry to the controller state:

            x = [x, y, z,
                phi, theta, psi,
                vx, vy, vz,
                p, q, r]

        Gazebo /gazebo/model_states provides both linear and angular
        velocities in the WORLD frame.

        The controller model uses:
            - linear velocity [vx, vy, vz] in WORLD frame
            - angular velocity [p, q, r] in BODY frame

        Therefore only angular velocity must be transformed:
            omega_body = R_WB.T @ omega_world
        """

        q = msg.pose.pose.orientation
        phi, theta, psi = quat_to_euler(q)

        # Rotation matrix: world <- body
        qx = q.x
        qy = q.y
        qz = q.z
        qw = q.w

        R_WB = np.array([
            [
                1.0 - 2.0 * (qy*qy + qz*qz),
                2.0 * (qx*qy - qz*qw),
                2.0 * (qx*qz + qy*qw),
            ],
            [
                2.0 * (qx*qy + qz*qw),
                1.0 - 2.0 * (qx*qx + qz*qz),
                2.0 * (qy*qz - qx*qw),
            ],
            [
                2.0 * (qx*qz - qy*qw),
                2.0 * (qy*qz + qx*qw),
                1.0 - 2.0 * (qx*qx + qy*qy),
            ],
        ])

        # Gazebo ModelStates twist is expressed in WORLD frame.
        v_world = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])

        omega_world = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ])

        # Convert angular velocity to BODY frame.
        omega_body = R_WB.T @ omega_world

        self.x = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,

            phi,
            theta,
            psi,

            v_world[0],
            v_world[1],
            v_world[2],

            omega_body[0],
            omega_body[1],
            omega_body[2],
        ])

        # ----------------------------------------------------
        # Odometry timing diagnostics
        # ----------------------------------------------------

        self.odom_seq += 1

        # ----------------------------------------------------
        # Odometry callback-rate diagnostic
        # ----------------------------------------------------

        now_diag = rospy.get_time()

        if (
            now_diag - self._odom_diag_last_time
            >= 1.0
        ):
            dt_diag = (
                now_diag
                - self._odom_diag_last_time
            )

            odom_cb_rate = (
                self.odom_seq
                - self._odom_diag_last_seq
            ) / dt_diag

            rospy.loginfo(
                "ODOM CALLBACK RATE: "
                "%.1f Hz seq=%d",
                odom_cb_rate,
                self.odom_seq
            )

            self._odom_diag_last_seq = (
                self.odom_seq
            )

            self._odom_diag_last_time = (
                now_diag
            )

        if msg.header.stamp.to_sec() > 0.0:
            self.odom_stamp = (
                msg.header.stamp.to_sec()
            )
        else:
            self.odom_stamp = None

        self.odom_arrival_time = rospy.get_time()

        self.have_odom = True

        # Wake the event-driven control loop (no-op for PID path).
        self._odom_event.set()

    def _ref_cb(self, msg):
        if len(msg.data) == 12:
            self.x_ref = np.array(msg.data)

    def control_law(self, x, x_ref):
        """Override: return u = [T, tphi, ttheta, tpsi]."""
        raise NotImplementedError

    def clamp_u(self, u):
        u[0] = np.clip(u[0], self.T_min, self.T_max)
        u[1:4] = np.clip(u[1:4], -self.tau_max, self.tau_max)
        du = np.clip(u - self.u_prev, -self.du_max, self.du_max)
        u = self.u_prev + du
        return u

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("[%s] waiting for odometry...", rospy.get_name())
        while not self.have_odom and not rospy.is_shutdown():
            rate.sleep()
        rospy.loginfo("[%s] odometry OK, controller active at %.0f Hz",
                      rospy.get_name(), self.rate_hz)
        while not rospy.is_shutdown():

            # ------------------------------------------------
            # Event-driven clock (MPC path):
            # run exactly ONE control cycle per fresh odometry
            # sample, so the real control period matches the
            # model sampling time Ts. If odometry stalls, hold
            # the last applied wrench.
            # ------------------------------------------------
            if self.require_fresh_odom:
                got = self._odom_event.wait(timeout=0.2)
                self._odom_event.clear()
                if not got:
                    w_hold = self.mixer.to_rotors(self.u_prev)
                    msg_hold = Float32MultiArray()
                    msg_hold.data = w_hold.tolist()
                    self.cmd_pub.publish(msg_hold)
                    continue

            t0 = time.perf_counter()

            # ------------------------------------------------
            # Snapshot odometry/control timing
            # ------------------------------------------------

            x_control = self.x.copy()

            odom_seq_control = self.odom_seq
            odom_stamp_control = self.odom_stamp
            odom_arrival_control = self.odom_arrival_time

            new_odom_count = (
                odom_seq_control
                - self.last_control_odom_seq
            )

            self.last_control_odom_seq = (
                odom_seq_control
            )

            now_ros = rospy.get_time()

            if odom_arrival_control is not None:
                odom_age_ms = (
                    1000.0
                    * (
                        now_ros
                        - odom_arrival_control
                    )
                )
            else:
                odom_age_ms = float("nan")

            if odom_stamp_control is not None:
                stamp_age_ms = (
                    1000.0
                    * (
                        now_ros
                        - odom_stamp_control
                    )
                )
            else:
                stamp_age_ms = float("nan")

            rospy.loginfo_throttle(
                0.5,
                "ODOM SYNC: "
                "ctrl_t=%.5f "
                "seq=%d "
                "new_odom=%d "
                "arrival_age_ms=%.3f "
                "stamp_age_ms=%.3f "
                "q=%.5f",
                now_ros,
                odom_seq_control,
                new_odom_count,
                odom_age_ms,
                stamp_age_ms,
                x_control[10]
            )

            u = self.control_law(
                x_control,
                self.x_ref.copy()
            )
            u = self.clamp_u(u)
            self.u_prev = u
            self.comp_ms = (time.perf_counter() - t0) * 1000.0
            w = self.mixer.to_rotors(u)
            msg = Float32MultiArray(); msg.data = w.tolist()
            self.cmd_pub.publish(msg)
            log = Float32MultiArray()
            log.data = u.tolist() + [self.gamma, self.r, self.comp_ms]
            self.log_pub.publish(log)

            # Event-driven controllers are clocked by _odom_event;
            # the Rate sleep is used only by the classic path (PID).
            if not self.require_fresh_odom:
                rate.sleep()
