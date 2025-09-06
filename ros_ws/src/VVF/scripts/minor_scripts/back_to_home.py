#!/usr/bin/env python3

import rospy
from tf.transformations import euler_from_quaternion
import numpy as np
import math
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class Back_to_home:
    def __init__(self):
        rospy.init_node('back_to_home')

        self.joints = np.zeros(6)  # Placeholder for joint positions [q1, q2, q3]
        self.joints_goal = np.array([0.7, -0.8, 1.1, 0.0, 0.0, 0.0])  # Goal positions for the joints
        self.goal_reached = False  # Flag to check if the goal is reached

        # Subscriber for joint states
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)

        # Publishers
        self.joint_1_pub = rospy.Publisher('/kinova_arm_joint_1_velocity_controller/command', Float64, queue_size=10)
        self.joint_2_pub = rospy.Publisher('/kinova_arm_joint_2_velocity_controller/command', Float64, queue_size=10)
        self.joint_3_pub = rospy.Publisher('/kinova_arm_joint_3_velocity_controller/command', Float64, queue_size=10)
        self.joint_4_pub = rospy.Publisher('/kinova_arm_joint_4_velocity_controller/command', Float64, queue_size=10)
        self.joint_5_pub = rospy.Publisher('/kinova_arm_joint_5_velocity_controller/command', Float64, queue_size=10)
        self.joint_6_pub = rospy.Publisher('/kinova_arm_joint_6_velocity_controller/command', Float64, queue_size=10)

        self.rate = rospy.Rate(10)

    def joint_states_callback(self, joints_msg):
        """
        Callback to update joint positions as a NumPy array.
        """
        self.joints = np.array([joints_msg.position[2], joints_msg.position[3], joints_msg.position[4], joints_msg.position[5], joints_msg.position[6], joints_msg.position[7]])

    def motion_to_home(self):
        """
        Compute and publish joint velocities to move the robot to its home position.
        """
        if self.goal_reached:
            rospy.loginfo("Robot already at the home position.")
            return
        # Proportional control to compute joint velocities
        kp = 1.0  # Proportional gain for controlling joint velocities
        joint_velocities = kp * (self.joints_goal - self.joints)

        # Scale velocities to respect maximum allowed velocity
        max_allowed_velocity = 0.15  # Maximum velocity (rad/s)
        scaling_factor = np.max(np.abs(joint_velocities) / max_allowed_velocity)

        if scaling_factor > 1:
            joint_velocities = joint_velocities / scaling_factor

        # Check if the robot is close enough to the goal
        if np.all(np.abs(self.joints_goal - self.joints) < 0.01):  # Threshold for "close enough"
            rospy.loginfo("Robot has reached the home position!")
            self.goal_reached = True
            self.publish_joints_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Stop all motion
        else:
            rospy.loginfo(f"Moving robot to home: Current position {self.joints}, Goal position {self.joints_goal}")
            self.publish_joints_velocity(joint_velocities)

    def publish_joints_velocity(self, joints_velocity):
        """
        Publish velocities to the arm joints.
        """
        joint1_float = Float64()
        joint2_float = Float64()
        joint3_float = Float64()
        joint4_float = Float64()
        joint5_float = Float64()
        joint6_float = Float64()

        joint1_float.data = joints_velocity[0]
        joint2_float.data = joints_velocity[1]
        joint3_float.data = joints_velocity[2]
        joint4_float.data = joints_velocity[3]
        joint5_float.data = joints_velocity[4]
        joint6_float.data = joints_velocity[5]

        self.joint_1_pub.publish(joint1_float)
        self.joint_2_pub.publish(joint2_float)
        self.joint_3_pub.publish(joint3_float)
        self.joint_4_pub.publish(joint4_float)
        self.joint_5_pub.publish(joint5_float)
        self.joint_6_pub.publish(joint6_float)


if __name__ == '__main__':
    try:
        motion = Back_to_home()
        while not rospy.is_shutdown():
            motion.motion_to_home()
            motion.rate.sleep()
    except rospy.ROSInterruptException:
        pass
