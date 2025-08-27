from NewObstaclesManager.kalman_filter import KalmanFilter
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import rospy
from NewObstaclesManager.geometry_algorithms import *


class Obstacle:
    def __init__(self, obstacle_id, points, centroid, dt=0.1):
        """
        Initialize an obstacle.

        Args:
            obstacle_id (int): Unique ID for the obstacle.
            points (np.array): Nx3 array of points in the cluster.
            centroid (np.array): Initial centroid position of the obstacle.
            dt (float): Time step for the Kalman filter.
        """
        self.obstacle_id = obstacle_id
        self.points = points
        self.centroid = centroid
        self.type = "static"  # Default type is static; updated dynamically
        self.velocity = np.zeros(3)  # Initialize velocity as zero
        self.geometry_type= "sphere" # Init geometry to sphere
        self.radius = 0
        self.bottom_point = None
        self.top_point = None


        # Kalman filter for position and velocity tracking
        self.kalman_filter = KalmanFilter(initial_position=centroid, dt=dt)
        self.last_update_time = None  # To track the time of the last update


    def update_points(self, points):
        """
        Update the obstacle with new points.
        """
        self.points = points
               
    def update_geometry(self):
        geometry_values = best_fit("BFGS", self.points)
        if geometry_values["type"] == "cylinder":
            self.geometry_type = "cylinder"
            self.radius = geometry_values["radius"]
            self.bottom_point = geometry_values["bottom_point"]
            self.top_point = geometry_values["top_point"]
        elif geometry_values["type"] == "sphere":
            self.geometry_type = "sphere"
            self.radius = geometry_values["radius"]

    def update_position(self, new_centroid):
        """
        Update the position using the Kalman filter.
        """
        current_time = rospy.Time.now()
        if self.last_update_time is not None:
            dt = (current_time - self.last_update_time).to_sec()
            self.kalman_filter.dt = dt
        self.last_update_time = current_time

        self.kalman_filter.predict()
        self.kalman_filter.update(new_centroid)
        self.centroid = self.kalman_filter.get_position()

    def update_velocity(self):
        """
        Update the velocity and classify the obstacle as static or dynamic.
        """
        self.velocity = self.kalman_filter.get_velocity()
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0.05:
            self.type = "dynamic"
            #self.update_geometry()
        else:
            self.type = "static" 

    def get_velocity(self):
        """
        Get the current velocity of the obstacle.
        """
        return self.velocity
    
    #################################################
    def predict_future_state(self, timestep, dt):
        """
        Predict the future state of the obstacle at a given timestep.
        
        Args:
            timestep (int): Step index (e.g., 0, 1, 2, ...)
            dt (float): Duration of each time step

        Returns:
            Obstacle: A new Obstacle object with predicted points, centroid, velocity, etc.
        """
        future_obstacle = Obstacle(
            obstacle_id=self.obstacle_id,
            points=self.points.copy(),
            centroid=self.centroid.copy(),  # Will update below
            dt=dt
        )

        total_dt = timestep * dt

        # Predict new centroid position using constant velocity model
        future_obstacle.centroid = self.centroid + self.velocity * total_dt

        # Shift all points based on centroid displacement
        displacement = future_obstacle.centroid - self.centroid
        future_obstacle.points = self.points + displacement

        # Copy other features
        future_obstacle.velocity = self.velocity.copy()
        future_obstacle.type = self.type
        future_obstacle.geometry_type = self.geometry_type
        future_obstacle.radius = self.radius
        future_obstacle.bottom_point = (
            self.bottom_point + displacement if self.bottom_point is not None else None
        )
        future_obstacle.top_point = (
            self.top_point + displacement if self.top_point is not None else None
        )

        return future_obstacle

