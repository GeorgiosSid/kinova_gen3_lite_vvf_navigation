import numpy as np

class KalmanFilter:
    def __init__(self, initial_position, dt=0.1, process_variance=1e-2, measurement_variance=1e-1):
        """
        Initialize the Kalman Filter for a single cluster.
        """
        self.dt = dt  # Time step
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # State: [x, y, z, vx, vy, vz]
        self.state = np.hstack((initial_position, np.zeros(3)))  # [x, y, z, 0, 0, 0]
        self.P = np.eye(6)  # Covariance matrix

        # State transition model
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt

        # Process noise covariance
        self.Q = process_variance * np.eye(6)

        # Measurement model
        self.H = np.eye(3, 6)

        # Measurement noise covariance
        self.R = measurement_variance * np.eye(3)

    def predict(self):
        """
        Predict the next state and update the covariance matrix.
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        Update the state using the measurement.
        """
        y = measurement - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_position(self):
        """
        Get the estimated position.
        """
        return self.state[:3]

    def get_velocity(self):
        """
        Get the estimated velocity.
        """
        return self.state[3:]
