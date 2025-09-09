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

from ObstaclesManager.obstacle import Obstacle


class ObstaclesProcessor:
    def __init__(self, control_points_manager=None):
        """
        control_points_manager: instance of ControlPoints (thread-safe manager) or None.
        """
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cp_manager = control_points_manager

        # Parameters
        self.control_points_distance_threshold = rospy.get_param("~cp_distance_threshold", 0.15)
        self.distance_threshold = rospy.get_param("~distance_threshold", 0.25)   # clustering match threshold
        self.ground_threshold = rospy.get_param("~ground_threshold", 0.1)        # ignore points below this height
        self.eps = rospy.get_param("~eps", 0.15)                                 # DBSCAN epsilon
        self.min_samples = rospy.get_param("~min_samples", 12)                   # DBSCAN min samples
        self.velocity_threshold = rospy.get_param("~velocity_threshold", 0.05)   # (unused here; obstacle object may use)

        # State
        self.cached_transform = None
        self.obstacles = {}
        self.next_cluster_id = 0  # Unique ID counter for obstacles

        # ROS I/O
        rospy.Subscriber("/velodyne_points", PointCloud2, self.velodyne_callback)
        self.obstacle_markers_pub = rospy.Publisher("/obstacle_markers", MarkerArray, queue_size=10)
        self.point_cloud_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=10)
        self.control_points_pub = rospy.Publisher("/control_points_obstaclesProcessor", MarkerArray, queue_size=10)

        rospy.loginfo("ObstaclesProcessor initialized.")

    # ----------------------
    # TF helpers
    # ----------------------
    def update_cached_transform(self, stamp=None):
        """
        Try to get transform at the given timestamp (preferred) or latest.
        """
        try:
            if stamp is not None:
                self.cached_transform = self.tf_buffer.lookup_transform(
                    "base_link", "velodyne", stamp, rospy.Duration(0.05)
                )
            else:
                self.cached_transform = self.tf_buffer.lookup_transform("base_link", "velodyne", rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Unable to update transform from 'velodyne' to 'base_link'.")
            self.cached_transform = None

    # ----------------------
    # Main point cloud callback
    # ----------------------
    def velodyne_callback(self, msg: PointCloud2):
        # Keep TF coherent with the message time
        if self.cached_transform is None:
            self.update_cached_transform(msg.header.stamp)
        if self.cached_transform is None:
            rospy.logwarn("No valid transform available. Skipping point cloud processing.")
            return

        try:
            # Transform cloud into base_link frame
            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, self.cached_transform)

            # Extract XYZ points and remove ground
            points = np.array(list(pc2.read_points(
                transformed_cloud, field_names=("x", "y", "z"), skip_nans=True
            )))
            points = points[points[:, 2] >= self.ground_threshold]

            # Visualize control points (if any)
            #self.publish_control_points()

            # Filter around control points (thread-safe snapshot)
            points = self.filter_points_near_control_points(points)

            if points.size == 0:
                return

            # Cluster
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            labels = db.labels_
            clusters = {cid: points[labels == cid] for cid in np.unique(labels) if cid != -1}

            # Track & publish
            self.track_obstacles(clusters)

        except Exception as e:
            rospy.logwarn(f"Error processing point cloud: {e}")

    # ----------------------
    # Tracking
    # ----------------------
    def track_obstacles(self, clusters):
        """
        Track obstacles by matching existing obstacles to detected clusters (Hungarian assignment).
        """
        current_centroids = {cid: np.mean(pts, axis=0) for cid, pts in clusters.items()}
        previous_ids = list(self.obstacles.keys())
        current_ids = list(current_centroids.keys())

        if len(previous_ids) == 0 and len(current_ids) == 0:
            return

        # Cost matrix (Euclidean distance)
        cost_matrix = np.zeros((len(previous_ids), len(current_ids)))
        for i, prev_id in enumerate(previous_ids):
            prev_position = self.obstacles[prev_id].centroid
            for j, curr_id in enumerate(current_ids):
                curr_position = current_centroids[curr_id]
                cost_matrix[i, j] = np.linalg.norm(prev_position - curr_position)

        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = [], []

        matched_prev_ids = set()
        matched_curr_ids = set()

        # Update matched
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.distance_threshold:
                prev_id = previous_ids[row]
                curr_id = current_ids[col]
                matched_prev_ids.add(prev_id)
                matched_curr_ids.add(curr_id)

                obstacle = self.obstacles[prev_id]
                obstacle.update_points(clusters[curr_id])
                obstacle.update_position(current_centroids[curr_id])
                obstacle.update_velocity()  # internal velocity estimate

        # Create new for unmatched current clusters
        for curr_id in set(current_ids) - matched_curr_ids:
            unique_id = self.next_cluster_id
            new_obstacle = Obstacle(
                obstacle_id=unique_id,
                points=clusters[curr_id],
                centroid=current_centroids[curr_id],
            )
            self.obstacles[unique_id] = new_obstacle
            self.next_cluster_id += 1

        # Remove unmatched previous obstacles
        for prev_id in set(previous_ids) - matched_prev_ids:
            del self.obstacles[prev_id]

        # Publish markers
        self.publish_obstacles()

    # ----------------------
    # Control points filtering
    # ----------------------
    def filter_points_near_control_points(self, points: np.ndarray) -> np.ndarray:
        """
        Keep points that are at least control_points_distance_threshold away from any active control point.
        If no control points manager is provided or no active points, returns the input points unchanged.
        """
        if self.cp_manager is None:
            return points

        cp_positions = self.cp_manager.get_active_positions_snapshot()
        if cp_positions.size == 0:
            return points

        # distances: (N_points, N_cp)
        distances = np.linalg.norm(points[:, None, :] - cp_positions[None, :, :], axis=2)
        min_distances = np.min(distances, axis=1)

        filtered_points = points[min_distances >= self.control_points_distance_threshold]
        #self.publish_filtered_point_cloud(filtered_points)
        return filtered_points

    # ----------------------
    # Publishers
    # ----------------------
    def publish_filtered_point_cloud(self, points: np.ndarray):
        """Convert numpy array to PointCloud2 message and publish it."""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"  # consistent with transformed cloud frame

        # Convert numpy points to a list of tuples (x, y, z)
        point_list = [tuple(map(float, p)) for p in points]

        # Define PointCloud2 fields (X, Y, Z)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_msg = pc2.create_cloud(header, fields, point_list)
        self.point_cloud_pub.publish(point_cloud_msg)

    def publish_control_points(self):
        """
        Publishes control points as a MarkerArray (small red spheres) for RViz.
        Uses a snapshot from the control points manager if available.
        """
        if self.cp_manager is None:
            return

        markers = MarkerArray()
        names_positions = self.cp_manager.get_active_points_snapshot()

        for i, (name, pos) in enumerate(names_positions):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = rospy.Time.now()
            m.ns = "control_points"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])

            # Small spheres
            m.scale.x = 0.04
            m.scale.y = 0.04
            m.scale.z = 0.04

            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0

            m.lifetime = rospy.Duration(0)
            m.frame_locked = False

            markers.markers.append(m)

        self.control_points_pub.publish(markers)

    def publish_obstacles(self):
        marker_array = MarkerArray()
        delete_marker_array = MarkerArray()
        active_marker_ids = set()

        for obstacle in self.obstacles.values():
            # Centroid sphere
            centroid_marker = Marker()
            centroid_marker.header.frame_id = "base_link"
            centroid_marker.header.stamp = rospy.Time.now()
            centroid_marker.ns = "obstacle_centroids"
            centroid_marker.id = obstacle.obstacle_id
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.pose.position.x = float(obstacle.centroid[0])
            centroid_marker.pose.position.y = float(obstacle.centroid[1])
            centroid_marker.pose.position.z = float(obstacle.centroid[2])
            centroid_marker.scale.x = 0.05
            centroid_marker.scale.y = 0.05
            centroid_marker.scale.z = 0.05
            centroid_marker.color.r = 0.0
            centroid_marker.color.g = 1.0 if obstacle.type == "static" else 0.0
            centroid_marker.color.b = 1.0 if obstacle.type == "dynamic" else 0.0
            centroid_marker.color.a = 1.0
            marker_array.markers.append(centroid_marker)
            active_marker_ids.add((centroid_marker.ns, centroid_marker.id))

            # Geometry markers
            if obstacle.geometry_type == "cylinder":
                cylinder_marker = Marker()
                cylinder_marker.header.frame_id = "base_link"
                cylinder_marker.header.stamp = rospy.Time.now()
                cylinder_marker.ns = "obstacle_cylinders"
                cylinder_marker.id = obstacle.obstacle_id + 1000
                cylinder_marker.type = Marker.CYLINDER
                cylinder_marker.action = Marker.ADD
                midpoint = (obstacle.bottom_point + obstacle.top_point) / 2.0
                height = float(np.linalg.norm(obstacle.top_point - obstacle.bottom_point))
                cylinder_marker.pose.position.x = float(midpoint[0])
                cylinder_marker.pose.position.y = float(midpoint[1])
                cylinder_marker.pose.position.z = float(midpoint[2])
                cylinder_marker.scale.x = float(obstacle.radius * 2.0)
                cylinder_marker.scale.y = float(obstacle.radius * 2.0)
                cylinder_marker.scale.z = height
                cylinder_marker.color.r = 1.0
                cylinder_marker.color.g = 0.5
                cylinder_marker.color.b = 0.0
                cylinder_marker.color.a = 0.4
                marker_array.markers.append(cylinder_marker)
                active_marker_ids.add((cylinder_marker.ns, cylinder_marker.id))

            elif obstacle.geometry_type == "sphere":
                sphere_marker = Marker()
                sphere_marker.header.frame_id = "base_link"
                sphere_marker.header.stamp = rospy.Time.now()
                sphere_marker.ns = "obstacle_spheres"
                sphere_marker.id = obstacle.obstacle_id + 1000
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                sphere_marker.pose.position.x = float(obstacle.centroid[0])
                sphere_marker.pose.position.y = float(obstacle.centroid[1])
                sphere_marker.pose.position.z = float(obstacle.centroid[2])
                diameter = float(obstacle.radius * 2.0)
                sphere_marker.scale.x = diameter
                sphere_marker.scale.y = diameter
                sphere_marker.scale.z = diameter
                sphere_marker.color.r = 0.0
                sphere_marker.color.g = 0.0
                sphere_marker.color.b = 1.0
                sphere_marker.color.a = 0.4
                marker_array.markers.append(sphere_marker)
                active_marker_ids.add((sphere_marker.ns, sphere_marker.id))

            # Text marker
            text_marker = Marker()
            text_marker.header.frame_id = "base_link"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "obstacle_text"
            text_marker.id = obstacle.obstacle_id + 2000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(obstacle.centroid[0])
            text_marker.pose.position.y = float(obstacle.centroid[1])
            text_marker.pose.position.z = float(obstacle.centroid[2] + 0.3)
            text_marker.scale.z = 0.1
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            velocity_magnitude = float(np.linalg.norm(obstacle.velocity))
            text_marker.text = f"ID: {obstacle.obstacle_id}\nVel: {velocity_magnitude:.2f} m/s"
            marker_array.markers.append(text_marker)
            active_marker_ids.add((text_marker.ns, text_marker.id))

            # Velocity arrow
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "base_link"
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.ns = "velocity_arrows"
            arrow_marker.id = obstacle.obstacle_id + 3000
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.points = [
                Point(float(obstacle.centroid[0]), float(obstacle.centroid[1]), float(obstacle.centroid[2])),
                Point(
                    float(obstacle.centroid[0] + obstacle.velocity[0]),
                    float(obstacle.centroid[1] + obstacle.velocity[1]),
                    float(obstacle.centroid[2] + obstacle.velocity[2]),
                ),
            ]
            arrow_marker.scale.x = 0.02  # shaft diameter
            arrow_marker.scale.y = 0.04  # head diameter
            arrow_marker.scale.z = 0.06  # head length
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

        # Publish deletion markers first, then current markers
        self.obstacle_markers_pub.publish(delete_marker_array)
        self.obstacle_markers_pub.publish(marker_array)


def main():
    rospy.init_node("ObstaclesProcessor")
    # Standalone run: no control points filtering/markers unless a manager is injected by another node.
    processor = ObstaclesProcessor(control_points_manager=None)
    rospy.spin()


if __name__ == "__main__":
    main()
