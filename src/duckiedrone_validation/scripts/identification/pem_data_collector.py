#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pem_data_collector.py

Raw identification-data collector for the Duckiedrone DD21.

Purpose
-------
Collect the state and the ACTUAL plant input for identification of

    dx[k+1] = A_PEM dx[k] + B_PEM du[k] + w[k]

using exactly the same state and input definitions as Physics MPC.

State
-----
x = [
    x, y, z,
    phi, theta, psi,
    vx, vy, vz,
    p, q, r
]

Frames:
    position            : WORLD
    Euler attitude      : body orientation in WORLD
    linear velocity     : WORLD
    angular velocity    : BODY

Input
-----
u = [
    delta_T,
    tau_phi,
    tau_theta,
    tau_psi
]

where

    delta_T = T - T_hover

The actual body wrench is reconstructed from /cmd_motors using
the TRUE plant parameters /dd21_plant/dd21.

Important
---------
The collector stores RAW asynchronous state/input streams.
No online interpolation or state-input pairing is performed.

Temporal alignment and resampling to Ts = 0.01 s are performed
offline before PEM estimation.

Author: Abdallah GHOUL
2026
"""

import csv
import json
import os
import threading
from datetime import datetime, timezone

import numpy as np
import rospy
import rospkg

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


# ============================================================
# Utility functions
# ============================================================

def quat_to_euler(q):
    """
    Quaternion -> roll, pitch, yaw.

    Same convention as ControllerBase.
    """

    x = q.x
    y = q.y
    z = q.z
    w = q.w

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)

    phi = np.arctan2(
        t0,
        t1
    )

    t2 = np.clip(
        2.0 * (w * y - z * x),
        -1.0,
        1.0
    )

    theta = np.arcsin(
        t2
    )

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)

    psi = np.arctan2(
        t3,
        t4
    )

    return phi, theta, psi


def rotation_world_from_body(q):
    """
    Rotation matrix

        R_WB : body -> world
    """

    qx = q.x
    qy = q.y
    qz = q.z
    qw = q.w

    return np.array([
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


# ============================================================
# PEM Data Collector
# ============================================================

class PEMDataCollector:

    def __init__(self):

        rospy.init_node(
            "pem_data_collector"
        )

        self.lock = threading.Lock()

        # ----------------------------------------------------
        # Topics
        # ----------------------------------------------------

        topics = rospy.get_param(
            "/topics"
        )

        self.odom_topic = rospy.get_param(
            "~odom_topic",
            topics.get(
                "odom",
                "/odom"
            )
        )

        self.motor_topic = rospy.get_param(
            "~motor_topic",
            topics.get(
                "cmd_motors",
                "/cmd_motors"
            )
        )

        self.control_log_topic = rospy.get_param(
            "~control_log_topic",
            "/validation/control_log"
        )

        # ----------------------------------------------------
        # TRUE plant parameters
        #
        # Identification must describe the actual simulated
        # plant, not a perturbed controller model.
        # ----------------------------------------------------

        if rospy.has_param(
            "/dd21_plant/dd21"
        ):

            plant = rospy.get_param(
                "/dd21_plant/dd21"
            )

            self.plant_param_source = (
                "/dd21_plant/dd21"
            )

        else:

            rospy.logwarn(
                "/dd21_plant/dd21 unavailable; "
                "falling back to /dd21."
            )

            plant = rospy.get_param(
                "/dd21"
            )

            self.plant_param_source = (
                "/dd21"
            )

        self.mass = float(
            plant["mass"]
        )

        self.gravity = float(
            plant["gravity"]
        )

        self.kf = float(
            plant["k_f"]
        )

        self.km = float(
            plant["k_m"]
        )

        self.dx = float(
            plant.get(
                "arm_dx",
                0.0775
            )
        )

        self.dy = float(
            plant.get(
                "arm_dy",
                0.1075
            )
        )

        self.w_max = float(
            plant["max_rotor_velocity"]
        )

        self.hover_thrust = (
            self.mass
            * self.gravity
        )

        self.Ts = float(
            rospy.get_param(
                "/mpc/Ts",
                0.01
            )
        )

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        package_path = (
            rospkg.RosPack()
            .get_path(
                "duckiedrone_validation"
            )
        )

        default_root = os.path.join(
            package_path,
            "data",
            "pem_identification"
        )

        output_root = rospy.get_param(
            "~output_root",
            default_root
        )

        default_run_name = (
            "pem_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        self.run_name = rospy.get_param(
            "~run_name",
            default_run_name
        )

        self.run_dir = os.path.join(
            output_root,
            self.run_name
        )

        os.makedirs(
            self.run_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        self.state_path = os.path.join(
            self.run_dir,
            "state_raw.csv"
        )

        self.input_path = os.path.join(
            self.run_dir,
            "input_raw.csv"
        )

        self.diag_path = os.path.join(
            self.run_dir,
            "control_diag_raw.csv"
        )

        self.exc_path = os.path.join(
            self.run_dir,
            "excitation_raw.csv"
        )

        self.exc_file = open(
            self.exc_path,
            "w",
            newline="",
            buffering=1
        )

        self.exc_writer = csv.writer(self.exc_file)

        self.exc_writer.writerow([
            "t_ros",
            "phase_code",
            "t_stable",
            "t_exc",
            "delta_T_exc",
            "tau_phi_exc",
            "tau_theta_exc",
            "tau_psi_exc",
        ])

        self.metadata_path = os.path.join(
            self.run_dir,
            "metadata.json"
        )

        self.state_file = open(
            self.state_path,
            "w",
            newline="",
            buffering=1
        )

        self.input_file = open(
            self.input_path,
            "w",
            newline="",
            buffering=1
        )

        self.diag_file = open(
            self.diag_path,
            "w",
            newline="",
            buffering=1
        )

        self.state_writer = csv.writer(
            self.state_file
        )

        self.input_writer = csv.writer(
            self.input_file
        )

        self.diag_writer = csv.writer(
            self.diag_file
        )

        # ----------------------------------------------------
        # Headers
        # ----------------------------------------------------

        self.state_writer.writerow([
            "t_ros",
            "t_arrival",
            "seq",

            "x",
            "y",
            "z",

            "phi",
            "theta",
            "psi",

            "vx",
            "vy",
            "vz",

            "p",
            "q",
            "r",
        ])

        self.input_writer.writerow([
            "t_ros",

            "w1",
            "w2",
            "w3",
            "w4",

            "T",
            "delta_T",

            "tau_phi",
            "tau_theta",
            "tau_psi",

            "rotor_zero",
            "rotor_max",
        ])

        self.diag_writer.writerow([
            "t_ros",

            "T_commanded",
            "delta_T_commanded",

            "tau_phi_commanded",
            "tau_theta_commanded",
            "tau_psi_commanded",

            "gamma",
            "r_tube",
            "compute_ms",
        ])

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        self.state_count = 0
        self.input_count = 0
        self.diag_count = 0
        self.exc_count = 0

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.metadata = {

            "run_name":
                self.run_name,

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "plant_parameter_source":
                self.plant_param_source,

            "mass":
                self.mass,

            "gravity":
                self.gravity,

            "k_f":
                self.kf,

            "k_m":
                self.km,

            "arm_dx":
                self.dx,

            "arm_dy":
                self.dy,

            "max_rotor_velocity":
                self.w_max,

            "T_hover":
                self.hover_thrust,

            "Ts_target":
                self.Ts,

            "state_topic":
                self.odom_topic,

            "input_topic":
                self.motor_topic,

            "diagnostic_topic":
                self.control_log_topic,
            "excitation_diagnostic_topic":
                "/validation/pem_excitation",
            
            "excitation_diag_is_identification_input":
                False,
            
            "identification_input_source":
                "/cmd_motors reconstructed body wrench",

            "state_order": [
                "x",
                "y",
                "z",
                "phi",
                "theta",
                "psi",
                "vx",
                "vy",
                "vz",
                "p",
                "q",
                "r"
            ],

            "input_order": [
                "delta_T",
                "tau_phi",
                "tau_theta",
                "tau_psi"
            ],

            "state_frames": {
                "position": "world",
                "linear_velocity": "world",
                "angular_velocity": "body"
            },

            "input_frame":
                "body",

            "input_definition":
                "delta_T = T - T_hover",

            "input_source":
                "reconstructed from /cmd_motors "
                "using true plant allocation parameters",

            "online_pairing":
                False
        }

        self._write_metadata()

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------

        rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_cb,
            queue_size=1,
            tcp_nodelay=True
        )

        rospy.Subscriber(
            self.motor_topic,
            Float32MultiArray,
            self.motor_cb,
            queue_size=1,
            tcp_nodelay=True
        )

        rospy.Subscriber(
            self.control_log_topic,
            Float32MultiArray,
            self.control_log_cb,
            queue_size=10
        )

        rospy.on_shutdown(
            self.shutdown
        )

        rospy.loginfo(
            "=== PEM Data Collector initialized ==="
        )

        rospy.loginfo(
            "State input : %s",
            self.odom_topic
        )

        rospy.loginfo(
            "Plant input : %s",
            self.motor_topic
        )

        rospy.loginfo(
            "T_hover     : %.6f N",
            self.hover_thrust
        )

        rospy.loginfo(
            "Output      : %s",
            self.run_dir
        )

        rospy.Subscriber(
            "/validation/pem_excitation",
            Float32MultiArray,
            self._excitation_cb,
            queue_size=100
        )

    # ========================================================

    def _write_metadata(self):

        with open(
            self.metadata_path,
            "w"
        ) as file:

            json.dump(
                self.metadata,
                file,
                indent=4
            )

    # ========================================================

    def odom_cb(
        self,
        msg
    ):
        """
        Record one raw state sample.
        """

        arrival_time = (
            rospy.Time.now()
            .to_sec()
        )

        if (
            msg.header.stamp.to_sec()
            > 0.0
        ):

            sample_time = (
                msg.header.stamp.to_sec()
            )

        else:

            sample_time = (
                arrival_time
            )

        q_msg = (
            msg.pose.pose.orientation
        )

        phi, theta, psi = (
            quat_to_euler(
                q_msg
            )
        )

        R_WB = (
            rotation_world_from_body(
                q_msg
            )
        )

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

        omega_body = (
            R_WB.T
            @ omega_world
        )

        row = [
            sample_time,
            arrival_time,
            msg.header.seq,

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
        ]

        with self.lock:

            self.state_writer.writerow(
                row
            )

            self.state_count += 1

    # ========================================================

    def motor_cb(
        self,
        msg
    ):
        """
        Reconstruct the actual BODY-frame wrench delivered
        to mixer_node from the final rotor commands.
        """

        if len(msg.data) != 4:

            rospy.logwarn_throttle(
                1.0,
                "PEM collector: expected "
                "4 rotor commands."
            )

            return

        sample_time = (
            rospy.Time.now()
            .to_sec()
        )

        w = np.asarray(
            msg.data,
            dtype=float
        ).reshape(4)

        w_sq = np.square(
            w
        )

        w1 = w_sq[0]
        w2 = w_sq[1]
        w3 = w_sq[2]
        w4 = w_sq[3]

        # ----------------------------------------------------
        # Exact forward allocation used by mixer_node
        # ----------------------------------------------------

        T = self.kf * (
            w1
            + w2
            + w3
            + w4
        )

        tau_phi = (
            self.kf
            * self.dy
            * (
                -w1
                - w2
                + w3
                + w4
            )
        )

        tau_theta = (
            self.kf
            * self.dx
            * (
                w1
                - w2
                + w3
                - w4
            )
        )

        tau_psi = (
            self.km
            * (
                w1
                - w2
                - w3
                + w4
            )
        )

        delta_T = (
            T
            - self.hover_thrust
        )

        rotor_zero = int(
            np.any(
                w <= 1.0e-6
            )
        )

        rotor_max = int(
            np.any(
                w >= (
                    self.w_max
                    - 1.0e-6
                )
            )
        )

        row = [
            sample_time,

            w[0],
            w[1],
            w[2],
            w[3],

            T,
            delta_T,

            tau_phi,
            tau_theta,
            tau_psi,

            rotor_zero,
            rotor_max,
        ]

        with self.lock:

            self.input_writer.writerow(
                row
            )

            self.input_count += 1

    # ========================================================

    def control_log_cb(
        self,
        msg
    ):
        """
        Optional diagnostic stream.

        /validation/control_log contains the physical wrench
        requested/applied by ControllerBase before rotor mixing.

        It is NOT used as the final identification input.
        """

        if len(msg.data) < 4:
            return

        sample_time = (
            rospy.Time.now()
            .to_sec()
        )

        T_cmd = float(
            msg.data[0]
        )

        delta_T_cmd = (
            T_cmd
            - self.hover_thrust
        )

        tau_phi = float(
            msg.data[1]
        )

        tau_theta = float(
            msg.data[2]
        )

        tau_psi = float(
            msg.data[3]
        )

        gamma = (
            float(msg.data[4])
            if len(msg.data) > 4
            else float("nan")
        )

        r_tube = (
            float(msg.data[5])
            if len(msg.data) > 5
            else float("nan")
        )

        compute_ms = (
            float(msg.data[6])
            if len(msg.data) > 6
            else float("nan")
        )

        with self.lock:

            self.diag_writer.writerow([
                sample_time,

                T_cmd,
                delta_T_cmd,

                tau_phi,
                tau_theta,
                tau_psi,

                gamma,
                r_tube,
                compute_ms,
            ])

            self.diag_count += 1

    # ========================================================

    def _excitation_cb(
        self,
        msg
    ):
        """
        Record PEM excitation-controller diagnostics.

        IMPORTANT
        ---------
        This stream is diagnostic only.

        It is NOT used as the identification input.

        The authoritative PEM input remains the actual BODY-frame
        wrench reconstructed from /cmd_motors.
        """

        if len(msg.data) < 7:

            rospy.logwarn_throttle(
                2.0,
                "PEM collector: invalid "
                "/validation/pem_excitation message."
            )

            return

        sample_time = (
            rospy.Time.now()
            .to_sec()
        )

        phase_code = float(
            msg.data[0]
        )

        t_stable = float(
            msg.data[1]
        )

        t_exc = float(
            msg.data[2]
        )

        delta_T_exc = float(
            msg.data[3]
        )

        tau_phi_exc = float(
            msg.data[4]
        )

        tau_theta_exc = float(
            msg.data[5]
        )

        tau_psi_exc = float(
            msg.data[6]
        )

        with self.lock:

            self.exc_writer.writerow([
                sample_time,
                phase_code,
                t_stable,
                t_exc,
                delta_T_exc,
                tau_phi_exc,
                tau_theta_exc,
                tau_psi_exc,
            ])

            self.exc_count += 1

    # =======================================================

    def shutdown(
        self
    ):

        with self.lock:

            self.metadata[
                "state_samples"
            ] = self.state_count

            self.metadata[
                "input_samples"
            ] = self.input_count

            self.metadata[
                "diagnostic_samples"
            ] = self.diag_count

            self.metadata[
                "excitation_samples"
            ] = self.exc_count

            self.metadata[
                "finished_utc"
            ] = datetime.now(
                timezone.utc
            ).isoformat()

            self._write_metadata()

            self.state_file.close()
            self.input_file.close()
            self.diag_file.close()
            self.exc_file.close()

        rospy.loginfo(
            "PEM data saved to %s",
            self.run_dir
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    collector = (
        PEMDataCollector()
    )

    rospy.spin()