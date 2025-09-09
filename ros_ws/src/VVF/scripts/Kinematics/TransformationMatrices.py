from Kinematics.constants import *
import math
"""
    "base_link" to "kinova_arm_base_link" : T_baselink_KinBaseLink

    "kinova_arm_base_link" to "kinova_arm_shoulder_link" : T_base_shoulder

    "kinova_arm_shoulder_link" to "kinova_arm_arm_link" : T_shoulder_arm

    "kinova_arm_arm_link" to "kinova_arm_forearm_link" : T_arm_forearm

    "kinova_arm_forearm_link" to "kinova_arm_lower_wrist_link" : T_forearm_lower_wrist

    "kinova_arm_lower_wrist_link" to "kinova_arm_upper_wrist_link" : T_lower_wrist_upper_wrist

    "kinova_arm_upper_wrist_link" to "kinova_arm_gripper_base_link" : T_upper_wrist_gripper_base

    "kinova_arm_gripper_base_link" to "kinova_arm_tool_frame" : T_upper_wrist_gripper_base
"""

def get_transformation_matrices(q1, q2, q3):
    """
    Returns all transformation matrices based on joint angles.
    :param q1: Joint angle 1 (radians)
    :param q2: Joint angle 2 (radians)
    :param q3: Joint angle 3 (radians)
    :return: Dictionary of transformation matrices
    """
    return {
        "T_baselink_KinBaseLink": [
            [1, 0, 0, m0],
            [0, 1, 0, 0],
            [0, 0, 1, m1],
            [0, 0, 0, 1],
        ],
        "T_base_shoulder": [
            [math.cos(q1), -math.sin(q1), 0, 0],
            [math.sin(q1), math.cos(q1), 0, 0],
            [0, 0, 1, l1],
            [0, 0, 0, 1],
        ],
        "T_shoulder_arm": [
            [math.cos(q2), -math.sin(q2), 0, 0],
            [0, 0, -1, -l2],
            [math.sin(q2), math.cos(q2), 0, l3],
            [0, 0, 0, 1],
        ],
        "T_shoulder_mid_arm": [
            [1, 0, 0, 0],
            [0, 1, 0, l4/2],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1],    
        ],
        "T_arm_forearm": [
            [math.cos(q3), -math.sin(q3), 0, 0],
            [-math.sin(q3), -math.cos(q3), 0, l4],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        "T_forearm_lower_wrist": [
            [1, 0, 0, 0],
            [0, 0, -1, -l5],
            [0, 1, 0, l6],
            [0, 0, 0, 1],
        ],
        "T_lower_wrist_upper_wrist": [
            [0, 0, 1, l7],
            [0, 1, 0, 0],
            [-1, 0, 0, l8],
            [0, 0, 0, 1],
        ],
        "T_upper_wrist_gripper_base": [
            [0, 0, -1, -l9],
            [0, 1, 0, 0],
            [1, 0, 0, l10],
            [0, 0, 0, 1],
        ],
        "T_gripper_base_tool_frame": [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, l11],
            [0, 0, 0, 1],
        ],
    }
