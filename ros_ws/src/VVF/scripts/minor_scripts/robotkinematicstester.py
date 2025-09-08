#!/usr/bin/env python3
import rospy
import numpy as np
import csv
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import tf
from tf.transformations import euler_from_quaternion
from Control_Points import ControlPoints  # My custom FK & Jacobian manager

class KinematicsValidation:
    def __init__(self):
        rospy.init_node("kinematics_validation")

        # Joint velocity publishers
        self.joint_1_pub = rospy.Publisher('/kinova_arm_joint_1_velocity_controller/command', Float64, queue_size=10)
        self.joint_2_pub = rospy.Publisher('/kinova_arm_joint_2_velocity_controller/command', Float64, queue_size=10)
        self.joint_3_pub = rospy.Publisher('/kinova_arm_joint_3_velocity_controller/command', Float64, queue_size=10)
        self.joint_4_pub = rospy.Publisher('/kinova_arm_joint_4_velocity_controller/command', Float64, queue_size=10)
        self.joint_5_pub = rospy.Publisher('/kinova_arm_joint_5_velocity_controller/command', Float64, queue_size=10)
        self.joint_6_pub = rospy.Publisher('/kinova_arm_joint_6_velocity_controller/command', Float64, queue_size=10)

        self.tf_listener = tf.TransformListener()
        self.controlPointsManager = ControlPoints()
        self.tool_frame_cp = next(cp for cp in self.controlPointsManager.control_points if cp.name == "tool_frame")

        # For smoothed velocity estimation
        self.position_buffer = []
        self.time_buffer = []
        self.window_size = 5

        # CSV Logging
        self.csv_file = open("kinematics_test_results.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Time", 
            "Joint Velocities", "Joint Velocities Computed", "Joint Velocity Error",
            "EE Position FK", "EE Position Real", "Position Error",
            "EE Velocity FK", "EE Velocity Real", "Velocity Error",
            "Orientation FK (RPY)", "Orientation Real (RPY)", "Orientation Error (rad)"
        ])

        self.test_joint_velocity = np.array([0.06, 0.02, 0.01])
        self.rate = rospy.Rate(10)
        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("Kinematics Validation Node Initialized.")

    def get_real_ee_pose(self):
        try:
            self.tf_listener.waitForTransform("base_link", "kinova_arm_tool_frame", rospy.Time(0), rospy.Duration(1.0))
            (pos, quat) = self.tf_listener.lookupTransform("base_link", "kinova_arm_tool_frame", rospy.Time(0))
            return np.array(pos), np.array(quat)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed.")
            return None, None

    def compute_smoothed_ee_velocity(self):
        if len(self.position_buffer) < 2:
            return np.zeros(3)

        times = np.array(self.time_buffer) - self.time_buffer[0]
        positions = np.array(self.position_buffer)

        velocities = []
        for i in range(3):
            slope, *_ = np.polyfit(times, positions[:, i], 1)
            velocities.append(slope)

        return np.array(velocities)

    def run_test(self):
        joint_angles = self.controlPointsManager.joints
        joint_velocities = self.controlPointsManager.joint_velocities

        ee_position_fk = self.tool_frame_cp.position
        ee_orientation_fk = self.tool_frame_cp.orientation
        ee_position_real, ee_orientation_real = self.get_real_ee_pose()

        if ee_position_real is None:
            return

        # --- Error Calculations ---
        position_error = np.linalg.norm(ee_position_fk - ee_position_real)
        # Convert quaternion to Euler angles (RPY)
        rpy_real = euler_from_quaternion(ee_orientation_real)  # (Roll, Pitch, Yaw)
        rpy_fk = euler_from_quaternion(ee_orientation_fk)  # (Roll, Pitch, Yaw)

        # Compute orientation error as norm of difference
        orientation_error_rpy = np.linalg.norm(np.array(rpy_real) - np.array(rpy_fk))

        # --- Velocity Estimation ---
        current_time = rospy.Time.now().to_sec()
        self.position_buffer.append(ee_position_real)
        self.time_buffer.append(current_time)
        if len(self.position_buffer) > self.window_size:
            self.position_buffer.pop(0)
            self.time_buffer.pop(0)

        ee_velocity_real = self.compute_smoothed_ee_velocity()
        J = self.tool_frame_cp.evaluate_jacobian(joint_angles)
        ee_velocity_fk = np.dot(J, joint_velocities)

        velocity_error = np.linalg.norm(ee_velocity_fk - ee_velocity_real)

        # --- Inverse Jacobian for Joint Velocities ---
        J_inv = np.linalg.pinv(J)
        joint_velocities_computed = np.dot(J_inv, ee_velocity_real)
        velocity_error_joint = np.linalg.norm(joint_velocities_computed - joint_velocities)

        # --- CSV Logging ---
        self.csv_writer.writerow([
            current_time,
            list(joint_velocities),
            list(joint_velocities_computed),
            velocity_error_joint,
            list(ee_position_fk),
            list(ee_position_real),
            position_error,
            list(ee_velocity_fk),
            list(ee_velocity_real),
            velocity_error,
            list(rpy_fk),
            list(rpy_real),
            orientation_error_rpy
        ])

        self.publish_joint_velocities()

    def publish_joint_velocities(self):
        self.joint_1_pub.publish(Float64(self.test_joint_velocity[0]))
        self.joint_2_pub.publish(Float64(self.test_joint_velocity[1]))
        self.joint_3_pub.publish(Float64(self.test_joint_velocity[2]))
        zero_float = Float64()
        zero_float.data = 0.0
        self.joint_4_pub.publish(zero_float)
        self.joint_5_pub.publish(zero_float)
        self.joint_6_pub.publish(zero_float)

    def shutdown_hook(self):
        rospy.loginfo("Shutting down. Closing CSV file.")
        self.csv_file.close()

if __name__ == "__main__":
    try:
        tester = KinematicsValidation()
        while not rospy.is_shutdown():
            tester.run_test()
            tester.rate.sleep()
    except rospy.ROSInterruptException:
        pass
