#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
import csv
import copy

from Control_Points import ControlPoints
from ObstaclesManager.obstaclesProcessor import ObstaclesProcessor
from Artificial_Functions.attractive_function import *
from Artificial_Functions.dynamicRepulsive import *
from Artificial_Functions.staticRepulsive import *
from minor_scripts.markersVisualization import *
from Inverse_Kinematics.inverse_kinematics_calc import *
from Kinematics.FK import *

class DynamicObstacleAvoidance:
    def __init__(self):
        rospy.init_node('DynamicObstacleAvoidance')

        self.goal_position = np.array([0.64, 0.15, 0.5])
        self.max_allowed_joint_velocity = 0.4

        # Initialize managers (single source of truth)
        self.controlPointsManager = ControlPoints()
        # NOTE: ObstaclesProcessor should accept the manager, not a raw list
        self.ObstaclesManager = ObstaclesProcessor(self.controlPointsManager)

        # Shortcuts to shared obstacle dict (owned by ObstaclesProcessor)
        self.obstacles = self.ObstaclesManager.obstacles

        # Publishers for joint velocities and markers
        self.joint_1_pub = rospy.Publisher('/kinova_arm_joint_1_velocity_controller/command', Float64, queue_size=10)
        self.joint_2_pub = rospy.Publisher('/kinova_arm_joint_2_velocity_controller/command', Float64, queue_size=10)
        self.joint_3_pub = rospy.Publisher('/kinova_arm_joint_3_velocity_controller/command', Float64, queue_size=10)
        self.joint_4_pub = rospy.Publisher('/kinova_arm_joint_4_velocity_controller/command', Float64, queue_size=10)
        self.joint_5_pub = rospy.Publisher('/kinova_arm_joint_5_velocity_controller/command', Float64, queue_size=10)
        self.joint_6_pub = rospy.Publisher('/kinova_arm_joint_6_velocity_controller/command', Float64, queue_size=10)

        self.attractive_force_marker_pub = rospy.Publisher('/attractive_force_marker', Marker, queue_size=10)
        self.repulsive_force_marker_pub = rospy.Publisher('/repulsive_force_markers', MarkerArray, queue_size=10)
        self.total_force_marker_pub = rospy.Publisher('/total_force_marker', Marker, queue_size=10)

        self.obstacles_point_marker_pub = rospy.Publisher("/obstacles_points", MarkerArray, queue_size=10)
        self.collision_point_marker_pub = rospy.Publisher("/collision_points", MarkerArray, queue_size=10)

        self.start_time = rospy.Time.now().to_sec()  # Initialize start time

        # CSV logging
        self.csv_file = open("robot_motion_data.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time", "joint_positions", "joint_velocities", "end_effector_position", "velocity_magnitude"])

        # IK targets
        self.update_goal_configurations()
        self.update_closer_configuration()
        self.controlPointsManager.update_control_points_goal_positions(self.goal_configuration)

        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("Dynamic Obstacle Avoidance initialized.")

        self.rate = rospy.Rate(10)
        self.prev_joint_velocities = np.zeros(3)
        self.max_joint_acceleration = 0.3  # rad/s²
        self.max_joint_deceleration = 0.6
        self.control_rate = 10  # Hz

    def dynamic_obstacle_avoidance(self):
        # Take a consistent snapshot from the control-points manager
        with self.controlPointsManager.lock():
            joints = self.controlPointsManager.joints.copy()
            joint_velocities_current = self.controlPointsManager.joint_velocities.copy()
            active_points = [cp for cp in self.controlPointsManager.control_points if cp.active]
            tool_frame_cp = next((cp for cp in active_points if cp.name == "tool_frame"), None)
            center_of_extention = self.controlPointsManager.center_of_extention.copy()

        if tool_frame_cp is None:
            rospy.logwarn("No active 'tool_frame' control point found; skipping this cycle.")
            return

        let_it_pass = False
        attractive_joints_vel = np.zeros(3)

        # ------------------------ Repulsive ------------------------
        marker_array = MarkerArray()
        marker_array_obstacle_points = MarkerArray()
        marker_array_collision_points = MarkerArray()
        marker_id = 0

        total_repulsive_joint_space = np.zeros(3)

        # Work on a snapshot of obstacles to avoid mutating while iterating
        obstacles_snapshot = copy.deepcopy(self.obstacles)

        for obstacle in obstacles_snapshot.values():
            if obstacle.type == "dynamic":
                if not let_it_pass:
                    let_it_pass = self.check_let_it_pass(obstacle)
                closest_cp, closest_obstacle_point, distance, cp_vel, collision_point = dynamic_repulsive(
                    obstacle, active_points
                )
                if closest_cp is not None:
                    cp_position = closest_cp.position.copy()
                    obstacle_velocity = obstacle.velocity
                    repulsive_cartesian_vel = compute_dynamic_repulsive(
                        cp_position, closest_obstacle_point, cp_vel, obstacle_velocity, distance
                    )

                    # Markers: collision and obstacle points, plus repulsive vector
                    marker = create_point_marker(collision_point, marker_id)
                    marker_array_collision_points.markers.append(marker)

                    marker = create_point_marker(closest_obstacle_point, marker_id)
                    marker_array_obstacle_points.markers.append(marker)

                    marker = create_force_marker(cp_position, repulsive_cartesian_vel, marker_id,
                                                 color=(1.0, 0.0, 0.0), length=0.15)
                    marker_array.markers.append(marker)
                    marker_id += 1

                    # Map Cartesian repulsion to joint-space using the CP's Jacobian
                    repulsive_joints_vel = closest_cp.compute_velocity_joints_space(joints, repulsive_cartesian_vel)
                    total_repulsive_joint_space += repulsive_joints_vel
                else:
                    obstacle.type = "static"

            if obstacle.type == "static":
                rospy.loginfo("static repulsive")
                repulsive_joint_vel, repulsive_cart_vel, cp_position, obstacle_point_pos = staticRepulsiveImproved(
                    joints, obstacle, active_points
                )
                if np.linalg.norm(repulsive_cart_vel) > 0:
                    marker = create_point_marker(obstacle_point_pos, marker_id)
                    marker_array_obstacle_points.markers.append(marker)

                    marker = create_force_marker(cp_position, repulsive_cart_vel, marker_id,
                                                 color=(1.0, 0.0, 0.0), length=0.15)
                    marker_array.markers.append(marker)
                    marker_id += 1
                total_repulsive_joint_space += repulsive_joint_vel

        # Optional: other repulsives (floor/workspace) you had commented out

        #self.obstacles_point_marker_pub.publish(marker_array_obstacle_points)
        self.repulsive_force_marker_pub.publish(marker_array)
        #self.collision_point_marker_pub.publish(marker_array_collision_points)

        # ------------------------ Attractive ------------------------
        current_time = rospy.Time.now().to_sec()
        elapsed_time = current_time - self.start_time
        if not let_it_pass:
            attractive_cart_vel = attractive_cartesian(self.goal_position, tool_frame_cp.position, elapsed_time)
            # attractive_marker = create_force_marker(tool_frame_cp.position, attractive_cart_vel, 100,
            #                                         color=(0.0, 1.0, 0.0), length=0.15)
            # self.attractive_force_marker_pub.publish(attractive_marker)
            attractive_joints_vel = tool_frame_cp.compute_velocity_joints_space(joints, attractive_cart_vel)
        else:
            rospy.loginfo("let it pass")

        rospy.loginfo(f"repulsive joints vel = {total_repulsive_joint_space}")

        # Combine
        joint_velocities = attractive_joints_vel + total_repulsive_joint_space

        # Cap overall joint velocity magnitude
        scaling_factor = np.max(np.abs(joint_velocities) / self.max_allowed_joint_velocity)
        if scaling_factor > 1:
            joint_velocities /= scaling_factor

        # Accel/decel limits
        joint_velocities = self.cap_joint_acceleration(joint_velocities, self.prev_joint_velocities)
        self.prev_joint_velocities = joint_velocities

        rospy.loginfo(f"total joints vel = {joint_velocities}")

        # Visualize total cartesian velocity from the tool frame
        total_cart_vel = tool_frame_cp.compute_velocity_cartesian(joints, joint_velocities)
        velocity_magnitude = np.linalg.norm(total_cart_vel)
        total_marker = create_force_marker(tool_frame_cp.position, total_cart_vel, marker_id,
                                           color=(0.0, 0.0, 1.0), length=0.1)
        self.total_force_marker_pub.publish(total_marker)

        # Log + publish
        self.csv_writer.writerow([current_time, joints.tolist(), joint_velocities_current.tolist(),
                                  tool_frame_cp.position.tolist(), velocity_magnitude])
        self.publish_joints_velocity(joint_velocities)

    def update_goal_configurations(self, tol=0.05, max_attempts=10):
        x_ee_goal, y_ee_goal, z_ee_goal = self.goal_position.tolist()
        goal_configurations = inverse_kinematics(x_ee_goal, y_ee_goal, z_ee_goal)

        if not goal_configurations:
            for _ in range(max_attempts):
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
        # Choose the closest configuration to current joints (snapshot)
        with self.controlPointsManager.lock():
            current_joints = self.controlPointsManager.joints.copy()
        distances = [np.linalg.norm(np.array(config) - current_joints) for config in self.goal_configurations]
        closer_configuration = self.goal_configurations[np.argmin(distances)]
        self.goal_configuration = closer_configuration
        rospy.loginfo(f"Closer configuration: q1 = {closer_configuration[0]:.4f}, "
                      f"q2 = {closer_configuration[1]:.4f}, q3 = {closer_configuration[2]:.4f}")

    def check_let_it_pass(self, dynamic_obstacle, threshold=0.2):
        distances = np.linalg.norm(dynamic_obstacle.points - self.goal_position, axis=1)
        return np.min(distances) < threshold

    def cap_joint_acceleration(self, q_dot_new, q_dot_prev):
        dt = 1.0 / self.control_rate
        delta = q_dot_new - q_dot_prev
        delta_norm = np.linalg.norm(delta)

        if delta_norm < 1e-6:
            return q_dot_new  # no significant change

        is_acceleration = np.dot(delta, q_dot_prev) >= 0
        if is_acceleration:
            rospy.loginfo("acceleration")
            max_delta = self.max_joint_acceleration * dt
        else:
            rospy.loginfo("decelaration")
            max_delta = self.max_joint_deceleration * dt

        if delta_norm > max_delta:
            delta = delta / delta_norm * max_delta  # scale

        return q_dot_prev + delta

    def publish_joints_velocity(self, joints_velocity):
        joint1_float = Float64()
        joint2_float = Float64()
        joint3_float = Float64()
        zero_float = Float64()

        joint1_float.data = float(joints_velocity[0])
        joint2_float.data = float(joints_velocity[1])
        joint3_float.data = float(joints_velocity[2])
        zero_float.data = 0.0

        self.joint_1_pub.publish(joint1_float)
        self.joint_2_pub.publish(joint2_float)
        self.joint_3_pub.publish(joint3_float)
        self.joint_4_pub.publish(zero_float)
        self.joint_5_pub.publish(zero_float)
        self.joint_6_pub.publish(zero_float)

    def shutdown_hook(self):
        rospy.loginfo("Shutting down. Closing the CSV file.")
        try:
            self.csv_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    try:
        avoidance = DynamicObstacleAvoidance()
        while not rospy.is_shutdown():
            avoidance.dynamic_obstacle_avoidance()
            avoidance.rate.sleep()
    except rospy.ROSInterruptException:
        pass
