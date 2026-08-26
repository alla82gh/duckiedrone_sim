#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mixer_node.py — allocation layer of Figure 4.1 (plant side).

Subscribes : topics/cmd_motors (std_msgs/Float32MultiArray, 4 rotor
             velocities [rad/s]) from the active controller.
             /gazebo/model_states (gazebo_msgs/ModelStates) for the
             current attitude (frame adapter, see below).
Publishes  : /dd21/wrench (geometry_msgs/Wrench) consumed by the
             libgazebo_ros_force plugin attached to base_link.
Service    : /enable_motors (std_srvs/SetBool) — disarmed -> zero wrench.

Parameters are read from /dd21_plant (TRUE plant parameters loaded by
spawn_dd21.launch), NOT from /dd21: in scenario S5 the controllers
deliberately use a perturbed model, while the plant must stay nominal.

Forward mixing (rectangular DD21 layout, matches
quadrotor_model.AllocationMixer):
    M1 rear-right (CW), M2 front-right (CCW),
    M3 rear-left (CCW), M4 front-left (CW)
    T     = kf (w1^2 + w2^2 + w3^2 + w4^2)
    tphi  = kf dy (-w1^2 - w2^2 + w3^2 + w4^2)
    ttheta= kf dx ( w1^2 - w2^2 + w3^2 - w4^2)
    tpsi  = km ( w1^2 - w2^2 - w3^2 + w4^2)

GAZEBO FRAME ADAPTER (sim-only — not needed on the real drone):
libgazebo_ros_force (noetic) applies the received wrench via
Link::AddForce/AddTorque, i.e. in the WORLD frame. The mixing above
naturally yields a BODY-frame wrench, so it is rotated by the current
attitude before publishing:  F_w = R(q) [0 0 T]^T,  tau_w = R(q) tau_b.
On the hardware DD21 the ESCs physically produce body-frame thrust and
this adapter simply does not exist — the rest of the chain
(controllers -> /cmd_motors -> mixer) is identical sim and real.

AUTHOR: Abdallah GHOUL 2026
"""
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Wrench
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import SetBool, SetBoolResponse


class MixerNode(object):
    def __init__(self):
        rospy.init_node("mixer_node")
        p = rospy.get_param("/dd21_plant/dd21")
        t = rospy.get_param("/dd21_plant/topics")
        g = rospy.get_param("/dd21_plant/gazebo")
        self.kf = p["k_f"]; self.km = p["k_m"]
        self.dx = p.get("arm_dx", 0.0775); self.dy = p.get("arm_dy", 0.1075)
        self.model_name = g.get("model_name", "duckiedrone")
        self.q = None                    # latest body orientation (world<-body)
        self.armed = False
        self.pub = rospy.Publisher(t.get("wrench", "/dd21/wrench"),
                                   Wrench, queue_size=1)
        rospy.Subscriber(t["cmd_motors"], Float32MultiArray,
                         self.cmd_cb, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self.states_cb, queue_size=1)
        rospy.Service(t.get("enable_motors", "/enable_motors"), SetBool,
                      self.arm_cb)
        rospy.loginfo("mixer: kf=%.3e km=%.2e dx=%.4f dy=%.4f model=%s — "
                      "call /enable_motors true to arm",
                      self.kf, self.km, self.dx, self.dy, self.model_name)

    def arm_cb(self, req):
        self.armed = bool(req.data)
        return SetBoolResponse(success=True,
                               message="armed" if self.armed else "disarmed")

    def states_cb(self, msg):
        try:
            i = msg.name.index(self.model_name)
        except ValueError:
            return
        self.q = msg.pose[i].orientation

    def _rot(self):
        """Rotation matrix (world <- body) from the latest quaternion."""
        if self.q is None:
            return np.eye(3)
        x, y, z, w = self.q.x, self.q.y, self.q.z, self.q.w
        return np.array([
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
        ])

    def cmd_cb(self, msg):
        if not self.armed or len(msg.data) != 4:
            w2 = np.zeros(4)
        else:
            w2 = np.square(np.asarray(msg.data, dtype=float))
        w1, w2_, w3, w4 = w2
        # body-frame wrench from the mixing law
        T  = self.kf * (w1 + w2_ + w3 + w4)
        tx = self.kf * self.dy * (-w1 - w2_ + w3 + w4)
        ty = self.kf * self.dx * ( w1 - w2_ + w3 - w4)
        tz = self.km * ( w1 - w2_ - w3 + w4)
        # world-frame adapter for libgazebo_ros_force (AddForce/AddTorque)
        R = self._rot()
        F_w   = R @ np.array([0.0, 0.0, T])
        tau_w = R @ np.array([tx, ty, tz])
        wrench = Wrench()
        wrench.force.x,  wrench.force.y,  wrench.force.z  = F_w
        wrench.torque.x, wrench.torque.y, wrench.torque.z = tau_w
        self.pub.publish(wrench)


if __name__ == "__main__":
    MixerNode()
    rospy.spin()