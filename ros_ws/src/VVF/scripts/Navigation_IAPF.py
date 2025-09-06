#!/usr/bin/env python3

import rospy
import numpy as np
import csv
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from Control_Points import ControlPoints
from ObstaclesManager.obstaclesProcessor import ObstaclesProcessor
from Artificial_Functions.attractive_function import *
from Artificial_Functions.improvedRepulsive import *
from Inverse_Kinematics.inverse_kinematics_calc import *


class VirtualVelocityMotion:
    def __init__(self):
        rospy.init_node('VirtualVelocityMotion')

        # Define goals
        self.goals_x = [0.7,  0.0]
        self.goals_y = [-0.25, 0.0]
        self.goals_z = [0.5, 0.0]
        self.goal_positions = 0
        self.goal_position = np.array([self.goals_x[0], self.goals_y[0], self.goals_z[0]])
        self.goal_configurations = []
        self.goal_configuration = np.zeros(3)
        self.max_allowed_velocity = 0.25

        self.joints = np.zeros(3)  # Placeholder for joint positions [q1, q2, q3]

        # Initialize control points
        self.controlPointsManager = ControlPoints()
        self.ObstaclesManager = ObstaclesProcessor(self.controlPointsManager.control_points)
        self.control_points = [cp for cp in self.controlPointsManager.control_points if cp.active]
        self.obstacles = self.ObstaclesManager.obstacles

        # Publishers for joint velocities
        self.joint_1_pub = rospy.Publisher('/kinova_arm_joint_1_velocity_controller/command', Float64, queue_size=10)
        self.joint_2_pub = rospy.Publisher('/kinova_arm_joint_2_velocity_controller/command', Float64, queue_size=10)
        self.joint_3_pub = rospy.Publisher('/kinova_arm_joint_3_velocity_controller/command', Float64, queue_size=10)
        self.joint_4_pub = rospy.Publisher('/kinova_arm_joint_4_velocity_controller/command', Float64, queue_size=10)
        self.joint_5_pub = rospy.Publisher('/kinova_arm_joint_5_velocity_controller/command', Float64, queue_size=10)
        self.joint_6_pub = rospy.Publisher('/kinova_arm_joint_6_velocity_controller/command', Float64, queue_size=10)

        self.attractive_force_marker_pub = rospy.Publisher('/attractive_force_marker', Marker, queue_size=10)
        self.repulsive_force_marker_pub = rospy.Publisher('/repulsive_force_markers', MarkerArray, queue_size=10)
        self.total_force_marker_pub = rospy.Publisher('/total_force_marker', Marker, queue_size=10)
        self.goal_marker_pub = rospy.Publisher('/goal_position_marker', Marker, queue_size=10)


        # CSV File for logging
        self.csv_file = open("robot_motion_data_IAPF.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time", "joint_positions", "joint_velocities", "end_effector_position", "end_effector_cartesian_vel", "velocity_magnitude"])

        
        self.update_goal_configurations()
        self.update_closer_configuration()
        self.controlPointsManager.update_control_points_goal_positions(self.goal_configuration)

        self.start_time = rospy.Time.now().to_sec()  # Initialize start time
        rospy.loginfo("Motion initialized")
        self.rate = rospy.Rate(10)

    def vvf(self):
        self.publish_goal_marker()
        tool_frame = next(point for point in self.control_points if point.name == "tool_frame")
        joints = self.controlPointsManager.joints
        

        # Compute attractive Cartesian velocity
        current_time = rospy.Time.now().to_sec()
        elapsed_time = current_time - self.start_time  # Calculate elapsed time
        attractive_cart_vel = attractive_cartesian(self.goal_position, tool_frame.position, elapsed_time)
        attractive_joints_vel = tool_frame.compute_velocity_joints_space(joints, attractive_cart_vel)

        # Compute repulsive velocity
        repulsive_joints, repulsive_markers = repulsive_joints_vel(
            obstacles=self.obstacles,
            control_points=self.control_points,
            joints=joints
        )

        # Combine velocities and scale
        joint_velocities = attractive_joints_vel + repulsive_joints

        scaling_factor = np.max(np.abs(joint_velocities) / self.max_allowed_velocity)

        if scaling_factor > 1:
            #rospy.loginfo("q = "+str(joints))
            #rospy.loginfo("scalling")
            joint_velocities = joint_velocities / scaling_factor

        # Publish scaled velocities
        self.publish_joints_velocity(joint_velocities)
        
        #end_effector_vel = tool_frame.compute_velocity_cartesian(joints, joint_velocities)
        #end_effector_vel_marker = self.force_marker(tool_frame.position, end_effector_vel)
        
        self.publish_attractive_force_marker(tool_frame.position, attractive_cart_vel)
        self.repulsive_force_marker_pub.publish(repulsive_markers)
        #self.total_force_marker_pub.publish(end_effector_vel_marker)

        # Log data to CSV
        timestamp = rospy.Time.now().to_sec()
        end_effector_position = tool_frame.position
        joint_velocities = self.controlPointsManager.joint_velocities
        end_effector_cartesian_vel = tool_frame.compute_velocity_cartesian(joints, joint_velocities)
        velocity_magnitude = np.linalg.norm(end_effector_cartesian_vel)
        end_effector_vel_marker = self.force_marker(tool_frame.position, end_effector_cartesian_vel)
        self.total_force_marker_pub.publish(end_effector_vel_marker)

        self.csv_writer.writerow([timestamp, joints.tolist(), joint_velocities.tolist(), end_effector_position.tolist(), end_effector_cartesian_vel.tolist(), velocity_magnitude])
        self.check_reach_goal(tool_frame.position)

    def publish_joints_velocity(self, joints_velocity):
        joint1_float = Float64()
        joint2_float = Float64()
        joint3_float = Float64()
        zero_float = Float64()

        joint1_float.data = joints_velocity[0]
        joint2_float.data = joints_velocity[1]
        joint3_float.data = joints_velocity[2]
        zero_float.data = 0.0

        self.joint_1_pub.publish(joint1_float)
        self.joint_2_pub.publish(joint2_float)
        self.joint_3_pub.publish(joint3_float)
        self.joint_4_pub.publish(zero_float)
        self.joint_5_pub.publish(zero_float)
        self.joint_6_pub.publish(zero_float)

    def check_reach_goal(self, tool_position):
        rospy.loginfo("dist left = "+str(np.linalg.norm(self.goal_position - tool_position)))
        if np.linalg.norm(self.goal_position - tool_position) < 0.01:
            rospy.loginfo('Position reached.')
            self.goal_positions += 1

            if self.goal_positions < len(self.goals_x) - 1:
                self.goal_position = np.array([self.goals_x[self.goal_positions],
                                               self.goals_y[self.goal_positions],
                                               self.goals_z[self.goal_positions]])
                self.update_goal_configurations()
                self.update_closer_configuration()
                self.controlPointsManager.update_control_points_goal_positions(self.goal_configuration)    
            else:
                rospy.loginfo("All goals reached!")
                self.publish_joints_velocity([0.0, 0.0, 0.0])
                self.csv_file.close()
                rospy.signal_shutdown("Path complete.")
    
    def update_goal_configurations(self, tol=0.05, max_attempts=10):
        x_ee_goal = self.goal_position[0]
        y_ee_goal = self.goal_position[1]
        z_ee_goal = self.goal_position[2]

        goal_configurations = inverse_kinematics(x_ee_goal, y_ee_goal, z_ee_goal)
        
        if not goal_configurations:
            for attempt in range(max_attempts):
                # Slightly perturb the target position within ±tol
                x_rand = x_ee_goal + np.random.uniform(-tol, tol)
                y_rand = y_ee_goal + np.random.uniform(-tol, tol)
                z_rand = z_ee_goal + np.random.uniform(-tol, tol)

                goal_configurations = inverse_kinematics(x_rand, y_rand, z_rand)

                if goal_configurations:
                    break
                
        if not goal_configurations:
            rospy.logwarn("No acceptable inverse kinematics solutions found")
            self.publish_joints_velocity([0.0, 0.0, 0.0])
            rospy.signal_shutdown("Shutdown.")

        self.goal_configurations = goal_configurations
    
    def update_closer_configuration(self):
        # Choose the closest configuration to current joints
        distances = [np.linalg.norm(np.array(config) - self.joints) for config in self.goal_configurations]
        closer_configuration = self.goal_configurations[np.argmin(distances)]
        self.goal_configuration =  closer_configuration
        print(f"Closer configuration: q1 = {closer_configuration[0]:.4f}, q2 = {closer_configuration[1]:.4f}, q3 = {closer_configuration[2]:.4f}")
        
    def publish_attractive_force_marker(self, position, force_vector):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "attractive_force"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.points.append(Point(position[0], position[1], position[2]))
        end_position = position + force_vector
        marker.points.append(Point(end_position[0], end_position[1], end_position[2]))

        marker.scale.x = 0.02
        marker.scale.y = 0.04
        marker.scale.z = 0.06

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        self.attractive_force_marker_pub.publish(marker)
      
    def publish_goal_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal_position"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = self.goal_position[0]
        marker.pose.position.y = self.goal_position[1]
        marker.pose.position.z = self.goal_position[2]

        marker.scale.x = 0.04  # Radius
        marker.scale.y = 0.04
        marker.scale.z = 0.04

        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Opacity

        self.goal_marker_pub.publish(marker)
    """
    def force_marker(self, position, force_vector):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "total_force"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.points.append(Point(position[0], position[1], position[2]))
        end_position = position + force_vector
        marker.points.append(Point(end_position[0], end_position[1], end_position[2]))

        marker.scale.x = 0.02
        marker.scale.y = 0.04
        marker.scale.z = 0.06

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        return marker
    """
    
    def normalize_vector(self, vector, length):
        """ Normalize the vector and scale it to a fixed length. """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])  # Avoid division by zero
        return (vector / norm) * length

    def publish_attractive_force_marker(self, position, force_vector):
        """ Publishes a visualization marker for the attractive force. """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "attractive_force"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Normalize and scale the force vector
        scaled_force = self.normalize_vector(force_vector, length=0.15)

        marker.points.append(Point(position[0], position[1], position[2]))
        end_position = position + scaled_force
        marker.points.append(Point(end_position[0], end_position[1], end_position[2]))

        marker.scale.x = 0.02
        marker.scale.y = 0.04
        marker.scale.z = 0.06

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        self.attractive_force_marker_pub.publish(marker)

    def force_marker(self, position, force_vector):
        """ Creates a visualization marker for the total force vector. """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "total_force"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Normalize and scale the force vector
        scaled_force = self.normalize_vector(force_vector, length=0.15)

        marker.points.append(Point(position[0], position[1], position[2]))
        end_position = position + scaled_force
        marker.points.append(Point(end_position[0], end_position[1], end_position[2]))

        marker.scale.x = 0.02
        marker.scale.y = 0.04
        marker.scale.z = 0.06

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8

        return marker

if __name__ == '__main__':
    try:
        navigator = VirtualVelocityMotion()
        while not rospy.is_shutdown():
            navigator.vvf()
            navigator.rate.sleep()
    except rospy.ROSInterruptException:
        pass
