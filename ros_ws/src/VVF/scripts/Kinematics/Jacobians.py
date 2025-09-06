import sympy as sp
from FK import forward_kinematics_symbolic
import rospy  # For logging in a ROS environment

class JacobianCalculator:
    def __init__(self):
        """
        Initialize the JacobianCalculator with symbolic joint variables and precomputed Jacobians.
        """
        self.q1, self.q2, self.q3 = sp.symbols('q1 q2 q3')
        # Get symbolic positions from FK module
        self.positions = forward_kinematics_symbolic(self.q1, self.q2, self.q3)
        # Compute Jacobians for each control point
        self.jacobians = self.compute_all_jacobians()
        rospy.loginfo("JacobianCalculator initialized with symbolic Jacobians.")

    def compute_all_jacobians(self):
        """
        Compute symbolic Jacobians for all control points.
        Returns:
            A dictionary mapping control point names to their symbolic Jacobian matrices.
        """
        jacobians = {}
        for point_name, position in self.positions.items():
            rospy.loginfo(f"Computing Jacobian for {point_name}.")
            jacobians[point_name] = self.compute_symbolic_jacobian(position)
        return jacobians

    def compute_symbolic_jacobian(self, position):
        """
        Compute the symbolic Jacobian matrix for a given position vector.
        Args:
            position: A list or tuple of symbolic expressions [x, y, z].
        Returns:
            A 3x3 symbolic Jacobian matrix.
        """
        J = sp.Matrix([
            [sp.diff(position[0], self.q1), sp.diff(position[0], self.q2), sp.diff(position[0], self.q3)],
            [sp.diff(position[1], self.q1), sp.diff(position[1], self.q2), sp.diff(position[1], self.q3)],
            [sp.diff(position[2], self.q1), sp.diff(position[2], self.q2), sp.diff(position[2], self.q3)]
        ])
        return J
