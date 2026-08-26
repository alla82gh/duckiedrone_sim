#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
odom_bridge.py — publishes ground-truth odometry for the Gazebo DD21.

Reads /gazebo/model_states, extracts the configured model pose/twist, and
republishes as nav_msgs/Odometry on the topic given by /topics/odom.
Use this while your model has no native odometry plugin; disable with
launch arg use_bridge:=false once the real /odom exists.

Timing fixes (2026-08-23, rev2):
  * gazebo_msgs/ModelStates has NO header field, so the arrival time
    rospy.Time.now() is used as the sample stamp.  With queue_size=1
    there is no backlog, so arrival time ~= true sample time (~1 ms).
  * deliberate downsampling to ~rate_hz (default 100 Hz = control Ts) instead of
    flooding the graph with 1000 Hz
  * queue_size=1 on subscriber and publisher — never accumulate
    stale ModelStates/Odometry in queues
  * guard against sim-time going backwards (scenario reset)
"""
import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry


class OdomBridge(object):
    def __init__(self):
        rospy.init_node("odom_bridge")
        self.model = rospy.get_param("/gazebo/model_name", "duckiedrone")
        topic = rospy.get_param("/topics/odom", "/odom")

        # Deliberate odometry downsampling (Fix C)
        self.rate_hz = float(rospy.get_param("~rate_hz", 100.0))
        self.min_dt = 1.0 / self.rate_hz
        self.last_pub_stamp = None

        self.pub = rospy.Publisher(topic, Odometry, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb,
                         queue_size=1)
        rospy.loginfo("odom_bridge: %s -> %s @ %.0f Hz",
                      self.model, topic, self.rate_hz)

    def cb(self, msg):
        if self.model not in msg.name:
            return

        # ModelStates carries no header: arrival time is the best
        # available sample time (queue_size=1 => no backlog).
        stamp = rospy.Time.now()

        # Downsample: at most one message per min_dt.
        # Note: if sim time jumps backwards (reset), dt < 0 and we
        # publish immediately instead of starving the stream.
        if self.last_pub_stamp is not None:
            dt = (stamp - self.last_pub_stamp).to_sec()
            if 0.0 <= dt < self.min_dt:
                return
        self.last_pub_stamp = stamp

        i = msg.name.index(self.model)
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = "world"
        o.child_frame_id = "base_link"
        o.pose.pose = msg.pose[i]
        o.twist.twist = msg.twist[i]
        self.pub.publish(o)


if __name__ == "__main__":
    OdomBridge()
    rospy.spin()
