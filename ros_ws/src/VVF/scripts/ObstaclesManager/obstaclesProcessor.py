#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
import tf2_sensor_msgs
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from ObstaclesManager.obstacle import Obstacle


class ObstaclesProcessor:
    def __init__(self, control_points=None):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.control_points = control_points
        # Parameters
        self.control_points_distance_threshold = 0.1
        self.distance_threshold = rospy.get_param("~distance_threshold", 0.25)  # Matching threshold for clustering
        self.ground_threshold = rospy.get_param("~ground_threshold", 0.1)  # Ignore points below this height
        self.eps = rospy.get_param("~eps", 0.15)  # DBSCAN epsilon parameter 13
        self.min_samples = rospy.get_param("~min_samples", 12)  # DBSCAN minimum samples parameter 18
        self.velocity_threshold = rospy.get_param("~velocity_threshold", 0.05)  # Static/dynamic classification threshold

        # State
        self.cached_transform = None
        self.obstacles = {}
        self.next_cluster_id = 0  # Unique ID counter for obstacles

        # ROS Subscribers and Publishers
        rospy.Subscriber("/velodyne_points", PointCloud2, self.velodyne_callback)
        self.obstacle_markers_pub = rospy.Publisher('/obstacle_markers', MarkerArray, queue_size=10)
        
        self.point_cloud_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=10)
        self.control_points_pub = rospy.Publisher("/control_points_obstaclesProcessor", MarkerArray, queue_size=10)

        rospy.loginfo("ObstaclesProcessor initialized.")

    def update_cached_transform(self):
        try:
            self.cached_transform = self.tf_buffer.lookup_transform("base_link", "velodyne", rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Unable to update transform from 'velodyne' to 'base_link'.")
            self.cached_transform = None

    def velodyne_callback(self, msg):
        if self.cached_transform is None:
            self.update_cached_transform()
        if self.cached_transform is None:
            rospy.logwarn("No valid transform available. Skipping point cloud processing.")
            return

        try:
            # Transform and filter point cloud
            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, self.cached_transform)
            points = np.array(list(pc2.read_points(transformed_cloud, field_names=("x", "y", "z"), skip_nans=True)))
            points = points[points[:, 2] >= self.ground_threshold]

            # Filter points near control points
            points = self.filter_points_near_control_points(points)

            if len(points) == 0:
                #rospy.loginfo("No points above ground threshold. Skipping.")
                return

            # Perform DBSCAN clustering
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            labels = db.labels_
            clusters = {cluster_id: points[labels == cluster_id] for cluster_id in np.unique(labels) if cluster_id != -1}


            # Update obstacles
            self.track_obstacles(clusters)

        except Exception as e:
            rospy.logwarn(f"Error processing point cloud: {e}")

    def track_obstacles(self, clusters):
        """
        Track obstacles by matching existing obstacles to detected clusters.
        """
        current_centroids = {cluster_id: np.mean(points, axis=0) for cluster_id, points in clusters.items()}
        previous_ids = list(self.obstacles.keys())
        current_ids = list(current_centroids.keys())

        # Cost matrix for matching obstacles to clusters
        cost_matrix = np.zeros((len(previous_ids), len(current_ids)))
        for i, prev_id in enumerate(previous_ids):
            prev_position = self.obstacles[prev_id].centroid
            for j, curr_id in enumerate(current_ids):
                curr_position = current_centroids[curr_id]
                cost_matrix[i, j] = np.linalg.norm(prev_position - curr_position)

        # Solve the assignment problem using the Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_prev_ids = set()
        matched_curr_ids = set()

        # Update matched obstacles
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.distance_threshold:
                prev_id = previous_ids[row]
                curr_id = current_ids[col]
                matched_prev_ids.add(prev_id)
                matched_curr_ids.add(curr_id)

                # Update the obstacle with new cluster data
                obstacle = self.obstacles[prev_id]
                obstacle.update_points(clusters[curr_id])
                obstacle.update_position(current_centroids[curr_id])
                obstacle.update_velocity()  # Update velocity after position

        for curr_id in set(current_ids) - matched_curr_ids:
            unique_id = self.next_cluster_id
            new_obstacle = Obstacle(
                obstacle_id=unique_id,
                points=clusters[curr_id],
                centroid=current_centroids[curr_id],
            )
            self.obstacles[unique_id] = new_obstacle
            self.next_cluster_id += 1  # Increment for the next unique ID


        # Remove obstacles for unmatched previous clusters
        for prev_id in set(previous_ids) - matched_prev_ids:
            del self.obstacles[prev_id]

        self.publish_obstacles()

    def filter_points_near_control_points(self, points):
        if self.control_points:
            # Extract positions from the list of ControlPoint objects
            control_points_array = np.array([point.position for point in self.control_points])
            # Compute distances between points and control points
            distances = np.linalg.norm(points[:, None, :] - control_points_array[None, :, :], axis=2)
            # Find the minimum distance for each point
            min_distances = np.min(distances, axis=1)
            # Filter points based on distance threshold
            filtered_points = points[min_distances >= self.control_points_distance_threshold]
            
            #self.publish_filtered_point_cloud(filtered_points)
            #self.publish_control_points()
            
            return filtered_points
        return points  # If no control points, return all points

    def publish_filtered_point_cloud(self, points):
        """ Convert numpy array to PointCloud2 message and publish it. """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"  # Change to your desired frame
        
        # Convert numpy points to a list of tuples (x, y, z)
        point_list = [tuple(point) for point in points]

        # Define PointCloud2 fields (X, Y, Z)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Create PointCloud2 message
        point_cloud_msg = pc2.create_cloud(header, fields, point_list)

        # Publish the message
        self.point_cloud_pub.publish(point_cloud_msg)
        
    def publish_control_points(self):
        """ Publishes control points as a MarkerArray (small red spheres) for RViz. """
        
        marker_array = MarkerArray()
        
        for i, control_point in enumerate(self.control_points):
            marker = Marker()
            marker.header.frame_id = "base_link"  # Change this to the appropriate frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "control_points"
            marker.id = i  # Unique ID for each marker
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set position
            marker.pose.position.x = control_point.position[0]
            marker.pose.position.y = control_point.position[1]
            marker.pose.position.z = control_point.position[2]

            # Set scale (radius = 0.02m → diameter = 0.04m)
            marker.scale.x = 0.04
            marker.scale.y = 0.04
            marker.scale.z = 0.04

            # Set color (Red, full opacity)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Alpha (1.0 = fully visible)

            # Other settings
            marker.lifetime = rospy.Duration(0)  # 0 = persistent
            marker.frame_locked = False

            # Add marker to array
            marker_array.markers.append(marker)

        # Publish the MarkerArray
        #self.control_points_pub.publish(marker_array)
        rospy.loginfo("Published control points as MarkerArray.")

    
    def publish_obstacles(self):
        marker_array = MarkerArray()
        delete_marker_array = MarkerArray()
        active_marker_ids = set()

        # Add current markers for all obstacles
        for obstacle in self.obstacles.values():
            # Sphere marker for the centroid
            centroid_marker = Marker()
            centroid_marker.header.frame_id = "base_link"
            centroid_marker.header.stamp = rospy.Time.now()
            centroid_marker.ns = "obstacle_centroids"
            centroid_marker.id = obstacle.obstacle_id
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.pose.position.x = obstacle.centroid[0]
            centroid_marker.pose.position.y = obstacle.centroid[1]
            centroid_marker.pose.position.z = obstacle.centroid[2]
            centroid_marker.scale.x = 0.05
            centroid_marker.scale.y = 0.05
            centroid_marker.scale.z = 0.05
            centroid_marker.color.r = 0.0
            centroid_marker.color.g = 1.0 if obstacle.type == "static" else 0.0
            centroid_marker.color.b = 1.0 if obstacle.type == "dynamic" else 0.0
            centroid_marker.color.a = 1.0
            marker_array.markers.append(centroid_marker)
            active_marker_ids.add((centroid_marker.ns, centroid_marker.id))

            # Geometry-specific markers
            if obstacle.geometry_type == "cylinder":
                # Cylinder marker for the obstacle
                cylinder_marker = Marker()
                cylinder_marker.header.frame_id = "base_link"
                cylinder_marker.header.stamp = rospy.Time.now()
                cylinder_marker.ns = "obstacle_cylinders"
                cylinder_marker.id = obstacle.obstacle_id + 1000
                cylinder_marker.type = Marker.CYLINDER
                cylinder_marker.action = Marker.ADD
                midpoint = (obstacle.bottom_point + obstacle.top_point) / 2
                height = np.linalg.norm(obstacle.top_point - obstacle.bottom_point)
                cylinder_marker.pose.position.x = midpoint[0]
                cylinder_marker.pose.position.y = midpoint[1]
                cylinder_marker.pose.position.z = midpoint[2]
                cylinder_marker.scale.x = obstacle.radius * 2
                cylinder_marker.scale.y = obstacle.radius * 2
                cylinder_marker.scale.z = height
                cylinder_marker.color.r = 1.0
                cylinder_marker.color.g = 0.5
                cylinder_marker.color.b = 0.0
                cylinder_marker.color.a = 0.4
                marker_array.markers.append(cylinder_marker)
                active_marker_ids.add((cylinder_marker.ns, cylinder_marker.id))
            elif obstacle.geometry_type == "sphere":
                # Sphere marker for the obstacle
                sphere_marker = Marker()
                sphere_marker.header.frame_id = "base_link"
                sphere_marker.header.stamp = rospy.Time.now()
                sphere_marker.ns = "obstacle_spheres"
                sphere_marker.id = obstacle.obstacle_id + 1000
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                sphere_marker.pose.position.x = obstacle.centroid[0]
                sphere_marker.pose.position.y = obstacle.centroid[1]
                sphere_marker.pose.position.z = obstacle.centroid[2]
                sphere_marker.scale.x = obstacle.radius * 2
                sphere_marker.scale.y = obstacle.radius * 2
                sphere_marker.scale.z = obstacle.radius * 2
                sphere_marker.color.r = 0.0
                sphere_marker.color.g = 0.0
                sphere_marker.color.b = 1.0
                sphere_marker.color.a = 0.4
                marker_array.markers.append(sphere_marker)
                active_marker_ids.add((sphere_marker.ns, sphere_marker.id))

            # Text marker for ID and velocity
            text_marker = Marker()
            text_marker.header.frame_id = "base_link"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "obstacle_text"
            text_marker.id = obstacle.obstacle_id + 2000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = obstacle.centroid[0]
            text_marker.pose.position.y = obstacle.centroid[1]
            text_marker.pose.position.z = obstacle.centroid[2] + 0.3
            text_marker.scale.z = 0.1
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            velocity_magnitude = np.linalg.norm(obstacle.velocity)
            text_marker.text = f"ID: {obstacle.obstacle_id}\nVel: {velocity_magnitude:.2f} m/s"
            marker_array.markers.append(text_marker)
            active_marker_ids.add((text_marker.ns, text_marker.id))

            # Arrow marker for velocity vector
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "base_link"
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.ns = "velocity_arrows"
            arrow_marker.id = obstacle.obstacle_id + 3000
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.points = [
                Point(obstacle.centroid[0], obstacle.centroid[1], obstacle.centroid[2]),
                Point(
                    obstacle.centroid[0] + obstacle.velocity[0],
                    obstacle.centroid[1] + obstacle.velocity[1],
                    obstacle.centroid[2] + obstacle.velocity[2]
                )
            ]
            arrow_marker.scale.x = 0.02  # Shaft diameter
            arrow_marker.scale.y = 0.04  # Head diameter
            arrow_marker.scale.z = 0.06  # Head length
            arrow_marker.color.r = 1.0
            arrow_marker.color.g = 0.0
            arrow_marker.color.b = 0.0
            arrow_marker.color.a = 1.0
            marker_array.markers.append(arrow_marker)
            active_marker_ids.add((arrow_marker.ns, arrow_marker.id))

        # Remove old markers
        for namespace, offset in [
            ("obstacle_centroids", 0),
            ("obstacle_cylinders", 1000),
            ("obstacle_text", 2000),
            ("velocity_arrows", 3000),
        ]:
            for obstacle in self.obstacles.values():
                marker_id = obstacle.obstacle_id + offset
                if (namespace, marker_id) not in active_marker_ids:
                    delete_marker = Marker()
                    delete_marker.header.frame_id = "base_link"
                    delete_marker.header.stamp = rospy.Time.now()
                    delete_marker.ns = namespace
                    delete_marker.id = marker_id
                    delete_marker.action = Marker.DELETE
                    delete_marker_array.markers.append(delete_marker)

        # Publish deletion markers first
        self.obstacle_markers_pub.publish(delete_marker_array)

        # Publish updated markers
        self.obstacle_markers_pub.publish(marker_array)


def main():
    rospy.init_node("ObstaclesProcessor")
    processor = ObstaclesProcessor()
    rospy.spin()


if __name__ == "__main__":
    main()
