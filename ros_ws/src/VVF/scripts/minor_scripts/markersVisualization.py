import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

def create_force_marker(position, force_vector, marker_id, color=(1.0, 0.0, 0.0), length=0.15):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "force_markes"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Normalize and scale the force vector
    scaled_force = normalize_vector(force_vector, length)

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
    marker.color.a = 1.0  
    return marker

def create_point_marker(point_position, marker_id):  
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "points_markers"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = point_position[0]
    marker.pose.position.y = point_position[1]
    marker.pose.position.z = point_position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.04
    marker.scale.y = 0.04
    marker.scale.z = 0.04
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 0.8

    return marker
    
def normalize_vector(vector, length):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])  # Avoid division by zero
    return (vector / norm) * length
