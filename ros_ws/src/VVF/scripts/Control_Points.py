#!/usr/bin/env python3
import rospy
import numpy as np
from threading import RLock
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from Kinematics.FK import forward_kinematics
from Kinematics.Jacobians import JacobianCalculator
from Kinematics.constants import *


class ControlPoint:
    def __init__(self, name, symbolic_jacobian, active=True, max_reachable_radius=0):
        self.name = name
        self.position = np.zeros(3)        # [x, y, z]
        self.orientation = np.zeros(4)     # quaternion (convention depends on FK output)
        self.symbolic_jacobian = symbolic_jacobian
        self.goal_position = np.zeros(3)
        self.active = active
        self.max_reachable_radius = max_reachable_radius

    def evaluate_jacobian(self, joints):
        """
        Evaluate the symbolic Jacobian numerically for the current joint positions.
        Returns:
            A 3x3 numpy array representing the evaluated Jacobian matrix.
        """
        J_numeric = np.array(
            self.symbolic_jacobian.evalf(
                subs={'q1': joints[0], 'q2': joints[1], 'q3': joints[2]}
            )
        ).astype(np.float64)
        return J_numeric

    def compute_velocity_cartesian(self, joints, joint_velocities):
        """
        Compute the Cartesian velocity using the evaluated Jacobian and joint velocities.
        Returns:
            Cartesian velocity as a numpy array [vx, vy, vz].
        """
        J = self.evaluate_jacobian(joints)
        return np.dot(J, joint_velocities)

    def compute_velocity_joints_space(self, joints, cartesian_velocity):
        """
        Map a Cartesian velocity to joint-space using (damped) pseudoinverse if needed.
        """
        J = self.evaluate_jacobian(joints)

        # Condition number for singularity detection
        cond_number = np.linalg.cond(J)

        if cond_number > 1e6:
            rospy.logwarn("Jacobian is near-singular! Using damped pseudo-inverse.")
            lambda_reg = 0.05
            J_damped = J.T @ np.linalg.inv(J @ J.T + lambda_reg * np.eye(J.shape[0]))
            joint_velocities = np.dot(J_damped, cartesian_velocity)

            # Debug info vs plain pseudo-inverse
            J_inv = np.linalg.pinv(J)
            joint_velocities_singular = np.dot(J_inv, cartesian_velocity)
            rospy.loginfo(f"magnitude without damping: {np.linalg.norm(joint_velocities_singular)}")
            rospy.loginfo(f"magnitude with damping:    {np.linalg.norm(joint_velocities)}")

            if np.isnan(np.linalg.norm(joint_velocities)):
                rospy.logwarn("Computed NaN joint velocities from damped inverse; returning zeros.")
                return np.zeros(3)
        else:
            J_inv = np.linalg.pinv(J)
            joint_velocities = np.dot(J_inv, cartesian_velocity)

        return joint_velocities


class ControlPoints:
    """
    Thread-safe manager that owns the live set of control points plus joints state.
    Other components should hold a reference to this manager and request snapshots
    instead of caching the raw list.
    """

    def __init__(self):
        # Synchronization
        self._lock = RLock()

        # Kinematics helpers
        self.jacobian_calculator = JacobianCalculator()

        # Core state
        self.joints = np.zeros(3)           # [q1, q2, q3]
        self.joint_velocities = np.zeros(3) # [dq1, dq2, dq3]
        self.center_of_extention = np.zeros(3)

        # ROS publishers (visualization)
        self.position_marker_pub = rospy.Publisher("/control_points_positions", MarkerArray, queue_size=10)
        self.goal_position_marker_pub = rospy.Publisher("/control_points_goal_positions", MarkerArray, queue_size=10)

        # Build control points (symbolic Jacobians)
        symbolic_jacobians = self.jacobian_calculator.jacobians
        self.control_points = [
            ControlPoint("arm_link",           symbolic_jacobians["arm_link"],           active=False),
            ControlPoint("arm_mid_link",       symbolic_jacobians["arm_mid_link"],       active=False),
            ControlPoint("forearm_link",       symbolic_jacobians["forearm_link"],       active=True),
            ControlPoint("lower_wrist_link",   symbolic_jacobians["lower_wrist_link"],   active=True),
            ControlPoint("upper_wrist_link",   symbolic_jacobians["upper_wrist_link"],   active=True),
            ControlPoint("gripper_base_link",  symbolic_jacobians["gripper_base_link"],  active=True),
            ControlPoint("tool_frame",         symbolic_jacobians["tool_frame"],         active=True),
        ]

        # Initialize reachable radii from a nominal configuration
        max_reach_joints = np.array([0.0, 0.0, 0.0])
        self.initialize_max_reachable_radius(max_reach_joints)

        # Subscribe to joint states
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)

        rospy.loginfo("Control points initialized")
        self.rate = rospy.Rate(10)

    # =========================
    # Safe snapshot accessors
    # =========================
    def lock(self):
        """Expose the re-entrant lock as a context manager for short atomic sections."""
        return self._lock

    def get_active_positions_snapshot(self):
        """Return an (N,3) array of active control point positions (copy)."""
        with self._lock:
            return np.array([cp.position.copy() for cp in self.control_points if cp.active])

    def get_active_points_snapshot(self):
        """Return list of (name, position_copy) for active points."""
        with self._lock:
            return [(cp.name, cp.position.copy()) for cp in self.control_points if cp.active]

    def get_joints_snapshot(self):
        """Return a copy of current joint positions."""
        with self._lock:
            return self.joints.copy()

    def get_joint_velocities_snapshot(self):
        """Return a copy of current joint velocities."""
        with self._lock:
            return self.joint_velocities.copy()

    def get_control_point_by_name(self, name):
        """Return (name, position_copy) for a specific active control point, or None."""
        with self._lock:
            for cp in self.control_points:
                if cp.active and cp.name == name:
                    return (cp.name, cp.position.copy())
        return None

    # =========================
    # ROS callbacks & updates
    # =========================
    def joint_states_callback(self, joints_msg: JointState):
        with self._lock:
            joint_positions = joints_msg.position
            joint_velocities = joints_msg.velocity

            # Map into q1,q2,q3 indices as in your original code
            q1 = joint_positions[2]
            q2 = joint_positions[3]
            q3 = joint_positions[4]
            self.joints = np.array([q1, q2, q3])

            q1_vel = joint_velocities[2]
            q2_vel = joint_velocities[3]
            q3_vel = joint_velocities[4]
            self.joint_velocities = np.array([q1_vel, q2_vel, q3_vel])

            # Update dependent state (positions/orientations)
            self.update_control_points_locked()

    def update_control_points(self):
        """Public method: acquires lock then refreshes CPs from current joints."""
        with self._lock:
            self.update_control_points_locked()

    def update_control_points_locked(self):
        """
        Internal: assumes self._lock is held.
        Refresh positions/orientations using forward kinematics and publish markers.
        """
        control_points_data = forward_kinematics(self.joints)

        # Update control point positions and orientations
        for point in self.control_points:
            if point.name in control_points_data:
                point_data = control_points_data[point.name]
                point.position = np.array(point_data["position"])
                point.orientation = np.array(point_data["quaternion"])

        # Publish visualizations based on the just-updated state
        #self.publish_positions_locked()
        #self.publish_goal_positions_locked()

    def update_control_points_goal_positions(self, goal_configuration):
        """Update the goal positions for each control point from a target configuration."""
        with self._lock:
            control_points_goal_data = forward_kinematics(goal_configuration)
            for point in self.control_points:
                if point.name in control_points_goal_data:
                    point_data = control_points_goal_data[point.name]
                    point.goal_position = np.array(point_data["position"])
        rospy.loginfo("Control points goal positions updated based on closest IK solution.")

    def initialize_max_reachable_radius(self, joints):
        """
        Initialize per-point max reachable radii using a reference configuration.
        """
        with self._lock:
            control_points_positions = forward_kinematics(joints)
            self.center_of_extention = control_points_positions["arm_link"]["position"]

            for point in self.control_points:
                if not point.active:
                    continue
                position = control_points_positions[point.name]["position"]
                radius = np.linalg.norm(self.center_of_extention - position)
                point.max_reachable_radius = radius
                rospy.loginfo(f"[{point.name}] Max reachable radius initialized to: {radius:.4f} m from center of extention")

    # =========================
    # Visualization publishers
    # =========================
    def publish_positions(self):
        with self._lock:
            self.publish_positions_locked()

    def publish_positions_locked(self):
        """
        Publish the positions of control points as a MarkerArray.
        Assumes self._lock is held.
        """
        marker_array = MarkerArray()
        for i, point in enumerate(self.control_points):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "control_points_positions"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point.position[0])
            marker.pose.position.y = float(point.position[1])
            marker.pose.position.z = float(point.position[2])
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.4
            marker_array.markers.append(marker)

        self.position_marker_pub.publish(marker_array)

    def publish_goal_positions(self):
        with self._lock:
            self.publish_goal_positions_locked()

    def publish_goal_positions_locked(self):
        """
        Publish goal positions as a MarkerArray.
        Assumes self._lock is held.
        """
        marker_array = MarkerArray()
        for i, point in enumerate(self.control_points):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "control_points_goal_positions"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point.goal_position[0])
            marker.pose.position.y = float(point.goal_position[1])
            marker.pose.position.z = float(point.goal_position[2])
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.4
            marker_array.markers.append(marker)

        self.goal_position_marker_pub.publish(marker_array)


def main():
    rospy.init_node("control_points_node")
    control_points = ControlPoints()
    rospy.spin()


if __name__ == "__main__":
    main()
