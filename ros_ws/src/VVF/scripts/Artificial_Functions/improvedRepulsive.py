import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def repulsive_joints_vel(obstacles, control_points, joints, obstacle_threshold=0.15, KR=0.00023):
    total_joint_velocities = np.zeros(3)
    marker_array = MarkerArray()
    marker_id = 0

    if not obstacles:
        #rospy.loginfo("No obstacles detected. Skipping repulsive velocity computation.")
        return total_joint_velocities, marker_array

    # Create a copy of the obstacles dictionary to avoid RuntimeError during iteration
    obstacles_copy = list(obstacles.values())

    for obstacle in obstacles_copy:
        closest_cp = None
        closest_distance = float('inf')

        # Find the closest control point
        for control_point in control_points:
            if control_point.name == "arm_mid_link":
                continue  # Skip the arm_mid_link control point

            cp_position = control_point.position
            distances = np.linalg.norm(obstacle.points - cp_position, axis=1)
            min_distance = np.min(distances)
            if min_distance < closest_distance:
                closest_distance = min_distance
                closest_cp = control_point

        if closest_cp and closest_distance < obstacle_threshold:
            # Compute repulsive force
            repulsive_force = compute_repulsive_force(
                obstacle.points, closest_cp, max_neighbors=1, KR=KR, threshold=obstacle_threshold
            )

            # Add a marker for the repulsive force vector
            marker = create_repulsive_force_marker(
                closest_cp.position, repulsive_force, marker_id, color=(1.0, 0.0, 0.0)
            )
            marker_array.markers.append(marker)
            marker_id += 1

            # Compute joint velocities using the control point's Jacobian
            try:
                joint_velocities = closest_cp.compute_velocity_joints_space(joints, repulsive_force)
                total_joint_velocities += joint_velocities
            except np.linalg.LinAlgError as e:
                rospy.logwarn(f"Jacobian error for {closest_cp.name}: {str(e)}")

    return total_joint_velocities, marker_array

def compute_repulsive_force(obstacle_points, control_point, max_neighbors, KR, threshold, n=1):
    """
    Compute the repulsive force exerted by nearby obstacle points on a control point.
    Args:
        obstacle_points: Array of obstacle points.
        control_point_position: Position of the control point as a numpy array.
        max_neighbors: Maximum number of neighboring obstacle points to consider.
        KR: Repulsive force gain.
        threshold: Distance threshold for applying repulsive forces.

    Returns:
        total_force: Total repulsive force as a numpy array.
    """
    control_point_position = control_point.position
    cp_goal_position = control_point.goal_position
    distances = np.linalg.norm(obstacle_points - control_point_position, axis=1)
    closest_indices = np.argsort(distances)[:max_neighbors]
    neighbor_points = obstacle_points[closest_indices]

    total_force = np.zeros(3)
    for point in neighbor_points:
        distance = np.linalg.norm(point - control_point_position)
        if distance < threshold:
            distance_to_cp_goal = np.linalg.norm(cp_goal_position - control_point_position)
            #force_magnitude = -KR * (1.0 / threshold - 1.0 / distance) / (distance ** 2)
            #direction = (control_point_position - point) / distance
            #force = force_magnitude * direction
            term1 = KR * ((1/distance) - (1/threshold)) * (1/(distance**2)) * (distance_to_cp_goal**n)
            direction1 = (control_point_position - point) / distance
            term2 = - (1/2) * KR * (((1/distance) - (1/threshold))**2) * n * (distance_to_cp_goal**(n-1))
            direction2 = (control_point_position - cp_goal_position) / distance_to_cp_goal
            total_force += term1*direction1 + term2*direction2

    return total_force
"""
def create_repulsive_force_marker(position, force_vector, marker_id, color=(1.0, 0.0, 0.0)):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "repulsive_forces"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Set the start and end points of the arrow
    start_point = Point(position[0], position[1], position[2])
    end_point = Point(
        position[0] + force_vector[0],#*0.15
        position[1] + force_vector[1],
        position[2] + force_vector[2]
    )
    marker.points.append(start_point)
    marker.points.append(end_point)

    # Set the scale of the arrow
    marker.scale.x = 0.02  # Shaft diameter
    marker.scale.y = 0.04  # Head diameter
    marker.scale.z = 0.06  # Head length

    # Set the color
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0  # Fully opaque

    return marker
"""
def create_repulsive_force_marker(position, force_vector, marker_id, color=(1.0, 0.0, 0.0)):
    """
    Creates a visualization marker for a repulsive force vector.
    Args:
        position: Start position of the force vector (numpy array).
        force_vector: The force vector to visualize (numpy array).
        marker_id: Unique ID for the marker.
        color: RGB color tuple for the marker.

    Returns:
        A Marker object for visualization.
    """
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "repulsive_forces"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Normalize and scale the force vector
    scaled_force = normalize_vector(force_vector, length=0.15)

    # Set the start and end points of the arrow
    start_point = Point(position[0], position[1], position[2])
    end_point = Point(
        position[0] + scaled_force[0],
        position[1] + scaled_force[1],
        position[2] + scaled_force[2]
    )
    marker.points.append(start_point)
    marker.points.append(end_point)

    # Set the scale of the arrow
    marker.scale.x = 0.02  # Shaft diameter
    marker.scale.y = 0.04  # Head diameter
    marker.scale.z = 0.06  # Head length

    # Set the color
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0  # Fully opaque

    return marker


def normalize_vector(vector, length):
    """ Normalize the vector and scale it to a fixed length. """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])  # Avoid division by zero
    return (vector / norm) * length
