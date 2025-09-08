from Kinematics.TransformationMatrices import*
import numpy as np
from scipy.spatial.transform import Rotation as R
import sympy as sp
from Kinematics.constants import *

def forward_kinematics(joints):
    """
    Compute forward kinematics for the given joint angles.
    :param joints: List of joint angles [q1, q2, q3]
    :return: Dictionary with positions and quaternions for each frame
    """
    q1, q2, q3 = joints

    # Get transformation matrices
    matrices = get_transformation_matrices(q1, q2, q3)

    # Compute cumulative transformation matrices for each frame
    T_baselink_KinBaseLink = np.array(matrices["T_baselink_KinBaseLink"])
    T_base_shoulder = T_baselink_KinBaseLink @ np.array(matrices["T_base_shoulder"])
    T_shoulder_arm = T_base_shoulder @ np.array(matrices["T_shoulder_arm"])
    T_shoulder_mid_arm = T_shoulder_arm @ np.array(matrices["T_shoulder_mid_arm"])
    T_arm_forearm = T_shoulder_arm @ np.array(matrices["T_arm_forearm"])
    T_forearm_lower_wrist = T_arm_forearm @ np.array(matrices["T_forearm_lower_wrist"])
    T_lower_wrist_upper_wrist = T_forearm_lower_wrist @ np.array(matrices["T_lower_wrist_upper_wrist"])
    T_upper_wrist_gripper_base = T_lower_wrist_upper_wrist @ np.array(matrices["T_upper_wrist_gripper_base"])
    T_tool_frame = T_upper_wrist_gripper_base @ np.array(matrices["T_gripper_base_tool_frame"])

    # Extract positions and quaternions
    def extract_position_and_quaternion(T):
        position = T[:3, 3]  # Extract translation vector
        rotation_matrix = T[:3, :3]  # Extract rotation matrix
        quaternion = R.from_matrix(rotation_matrix).as_quat()  # Convert to quaternion
        return position, quaternion

    # Compute positions and quaternions for each frame
    frames = {
        "arm_link": extract_position_and_quaternion(T_shoulder_arm),
        "arm_mid_link": extract_position_and_quaternion(T_shoulder_mid_arm),
        "forearm_link": extract_position_and_quaternion(T_arm_forearm),
        "lower_wrist_link": extract_position_and_quaternion(T_forearm_lower_wrist),
        "upper_wrist_link": extract_position_and_quaternion(T_lower_wrist_upper_wrist),
        "gripper_base_link": extract_position_and_quaternion(T_upper_wrist_gripper_base),
        "tool_frame": extract_position_and_quaternion(T_tool_frame),
    }

    # Format the result to include positions and quaternions separately
    return {
        frame: {
            "position": position,
            "quaternion": quaternion
        }
        for frame, (position, quaternion) in frames.items()
    }

def forward_kinematics_symbolic(q1, q2, q3):
    x0 = l2 * sp.sin(q1) + m0
    y0 = -l2 * sp.cos(q1)
    z0 = m1 + l1 + l3

    x1 = 0.06 * sp.sin(q1) - 0.12 * sp.cos(q1) * sp.sin(q2) + 0.12
    y1 = -0.06 * sp.cos(q1) - 0.12 * sp.sin(q1) * sp.sin(q2)
    z1 = 0.12 * sp.cos(q2) + 0.4273

    x2 = x0 - l4 * sp.cos(q1) * sp.sin(q2)
    y2 = y0 - l4 * sp.sin(q1) * sp.sin(q2)
    z2 = l4 * sp.cos(q2) + z0

    x3 = -l5 * sp.sin(q2 - q3) * sp.cos(q1) - l6 * sp.sin(q1) + x2
    y3 = -l5 * sp.sin(q2 - q3) * sp.sin(q1) + l6 * sp.cos(q1) + y2
    z3 = l5 * sp.cos(q2 - q3) + z2

    x4 = sp.cos(q1) * (l7 * sp.cos(q2 - q3) - l8 * sp.sin(q2 - q3)) + x3
    y4 = sp.sin(q1) * (l7 * sp.cos(q2 - q3) - l8 * sp.sin(q2 - q3)) + y3
    z4 = l7 * sp.sin(q2 - q3) + l8 * sp.cos(q2 - q3) + z3

    x5 = sp.cos(q1) * (l10 * sp.cos(q2 - q3) - l9 * sp.sin(q2 - q3)) + x4
    y5 = sp.sin(q1) * (l10 * sp.cos(q2 - q3) - l9 * sp.sin(q2 - q3)) + y4
    z5 = l10 * sp.sin(q2 - q3) + l9 * sp.cos(q2 - q3) + z4

    x6 = x5 - l11 * sp.sin(q2 - q3) * sp.cos(q1)
    y6 = y5 - l11 * sp.sin(q2 - q3) * sp.sin(q1)
    z6 = z5 + l11 * sp.cos(q2 - q3)

    return {
        "arm_link": [x0, y0, z0],
        "arm_mid_link": [x1, y1, z1],
        "forearm_link": [x2, y2, z2],
        "lower_wrist_link": [x3, y3, z3],
        "upper_wrist_link": [x4, y4, z4],
        "gripper_base_link": [x5, y5, z5],
        "tool_frame": [x6, y6, z6]
    }
