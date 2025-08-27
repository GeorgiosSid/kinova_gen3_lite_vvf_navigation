import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, initial_position, dt=0.1, process_variance=1e-2, measurement_variance=1e-1):
        """
        Initialize the Extended Kalman Filter for position, velocity, and acceleration.
        """
        self.dt = dt  # Time step
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # State: [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.hstack((initial_position, np.zeros(6)))  # [pos, vel, accel]
        self.P = np.eye(9)  # Covariance matrix

        # State transition model
        self.F = np.eye(9)
        for i in range(3):
            self.F[i, i + 3] = dt
            self.F[i, i + 6] = 0.5 * dt**2
            self.F[i + 3, i + 6] = dt

        # Process noise covariance
        self.Q = process_variance * np.eye(9)

        # Measurement model
        self.H = np.zeros((3, 9))  # Only position is observed
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1

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
        self.P = (np.eye(9) - K @ self.H) @ self.P

    def get_position(self):
        """
        Get the estimated position.
        """
        return self.state[:3]

    def get_velocity(self):
        """
        Get the estimated velocity.
        """
        return self.state[3:6]

    def get_acceleration(self):
        """
        Get the estimated acceleration.
        """
        return self.state[6:]
