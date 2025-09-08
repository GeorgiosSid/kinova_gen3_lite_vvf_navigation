import numpy as np
import rospy

def staticRepulsive(joints, obstacle, control_points, obstacle_threshold=0.15, KR=0.00023):
    closest_control_point = None
    closest_obstacle_point = None
    closest_distance = float('inf')
    
    for control_point in control_points:
    
        control_point_position = control_point.position
        
        # We compute distances from control point to all obstacle points
        distances = np.linalg.norm(obstacle.points - control_point_position, axis=1)
        
        # We get the index of the closest obstacle point
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        
        # We update the closest values if a smaller distance is found
        if min_distance < closest_distance:
            closest_distance = min_distance
            closest_control_point = control_point
            closest_obstacle_point = obstacle.points[min_index]
    
    control_point_position = closest_control_point.position
    
    if closest_distance >= obstacle_threshold:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    else:    
        vel_magnitude = -KR * (1.0 / obstacle_threshold - 1.0 / closest_distance) / (closest_distance ** 2)
        direction = (control_point_position - closest_obstacle_point) / closest_distance
        repulsive_cart_vel = vel_magnitude * direction
        
        repulsive_joint_vel = closest_control_point.compute_velocity_joints_space(joints, repulsive_cart_vel)
    
        return repulsive_joint_vel, repulsive_cart_vel, control_point_position, closest_obstacle_point
    
            
def staticRepulsiveImproved(joints, obstacle, control_points, obstacle_threshold=0.15, KR=0.00023, n=1):
    closest_control_point = None
    closest_obstacle_point = None
    closest_distance = float('inf')
    
    for control_point in control_points:
            
        control_point_position = control_point.position
        
        # We compute distances from control point to all obstacle points
        distances = np.linalg.norm(obstacle.points - control_point_position, axis=1)
        
        # We get the index of the closest obstacle point
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        
        # We update the closest values if a smaller distance is found
        if min_distance < closest_distance:
            closest_distance = min_distance
            closest_control_point = control_point
            closest_obstacle_point = obstacle.points[min_index]
    
    control_point_position = closest_control_point.position
    
    if closest_distance >= obstacle_threshold:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    else:    
        cp_goal_position = control_point.goal_position
        distance_to_cp_goal = np.linalg.norm(cp_goal_position - control_point_position)
        
        term1 = KR * ((1/closest_distance) - (1/obstacle_threshold)) * (1/(closest_distance**2)) * (distance_to_cp_goal**n)
        direction1 = (control_point_position - closest_obstacle_point) / closest_distance
        term2 = - (1/2) * KR * (((1/closest_distance) - (1/obstacle_threshold))**2) * n * (distance_to_cp_goal**(n-1))
        direction2 = (control_point_position - cp_goal_position) / distance_to_cp_goal
        repulsive_cart_vel = term1*direction1 + term2*direction2
        
        repulsive_joint_vel = closest_control_point.compute_velocity_joints_space(joints, repulsive_cart_vel)
    
        return repulsive_joint_vel, repulsive_cart_vel, control_point_position, closest_obstacle_point
    
def staticRepulsiveFloor(joints, control_points, floor_z=0.0, obstacle_threshold=0.15, KR=0.00025):
    closest_control_point = None
    closest_distance = float('inf')

    for control_point in control_points:
        control_point_position = control_point.position
        distance_to_floor = abs(control_point_position[2] - floor_z)  # Distance in z-direction

        if distance_to_floor < closest_distance:
            closest_distance = distance_to_floor
            closest_control_point = control_point

    if closest_control_point is None:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    control_point_position = closest_control_point.position
    closest_floor_point = np.array([control_point_position[0], control_point_position[1], floor_z])

    # If the control point is above the threshold, no repulsion is needed
    if closest_distance >= obstacle_threshold:
        return np.zeros(3), np.zeros(3), control_point_position, closest_floor_point
    else:
        # Compute the repulsive velocity in the z-direction (pushing upwards)
        vel_magnitude = -KR * (1.0 / obstacle_threshold - 1.0 / closest_distance) / (closest_distance ** 2)
        direction = np.array([0, 0, 1])  # Upward direction in the z-axis
        repulsive_cart_vel = vel_magnitude * direction

        # Compute joint space repulsion (convert Cartesian velocity to joint velocity)
        repulsive_joint_vel = closest_control_point.compute_velocity_joints_space(joints, repulsive_cart_vel)

        return repulsive_joint_vel, repulsive_cart_vel, control_point_position, closest_floor_point
    
def workspace_boundary_repulsive(joints, control_points, center_of_extention, threshold=0.08, KR=0.0005):
    closest_control_point = None
    closest_distance = float('inf')

    for control_point in control_points:
        control_point_position = control_point.position
        distance_to_full_extend = abs(control_point.max_reachable_radius - np.linalg.norm(control_point_position - center_of_extention))
        
        if distance_to_full_extend < closest_distance:
            closest_distance = distance_to_full_extend
            closest_control_point = control_point
            
    control_point_position = closest_control_point.position
    
    direction = control_point_position - center_of_extention
    direction_norm = np.linalg.norm(direction)

    closest_workspace_boundary_point = center_of_extention + (direction / direction_norm) * closest_control_point.max_reachable_radius
    
    # If the control point is above the threshold, no repulsion is needed
    if closest_distance >= threshold:
        return np.zeros(3), np.zeros(3), control_point_position, closest_workspace_boundary_point
    else:
        vel_magnitude = -KR * (1.0 / threshold - 1.0 / closest_distance) / (closest_distance ** 2) 
        direction_of_repulsive = (control_point_position - closest_workspace_boundary_point) / closest_distance
        repulsive_cart_vel = vel_magnitude * direction_of_repulsive

        # Compute joint space repulsion (convert Cartesian velocity to joint velocity)
        repulsive_joint_vel = closest_control_point.compute_velocity_joints_space(joints, repulsive_cart_vel)

        return repulsive_joint_vel, repulsive_cart_vel, control_point_position, closest_workspace_boundary_point 
    

def workspace_boundary_repulsive_Improved(joints, control_points, center_of_extention, threshold=0.05, KR=0.0001, n=1):
    closest_control_point = None
    closest_distance = float('inf')

    for control_point in control_points:
        if control_point.name != "tool_frame":
            continue
        
        control_point_position = control_point.position
        distance_to_full_extend = abs(control_point.max_reachable_radius - np.linalg.norm(control_point_position - center_of_extention))
        
        if distance_to_full_extend < closest_distance:
            closest_distance = distance_to_full_extend
            closest_control_point = control_point
            
    control_point_position = closest_control_point.position
    
    direction = control_point_position - center_of_extention
    direction_norm = np.linalg.norm(direction)

    closest_workspace_boundary_point = center_of_extention + (direction / direction_norm) * closest_control_point.max_reachable_radius
    
    # If the control point is above the threshold, no repulsion is needed
    if closest_distance >= threshold:
        return np.zeros(3), np.zeros(3), control_point_position, closest_workspace_boundary_point
    else:
        rospy.loginfo(f" cp: {closest_control_point.name} with distance to boundary {closest_distance}")
        cp_goal_position = control_point.goal_position
        distance_to_cp_goal = np.linalg.norm(cp_goal_position - control_point_position)
        
        term1 = KR * ((1/closest_distance) - (1/threshold)) * (1/(closest_distance**2)) * (distance_to_cp_goal**n)
        direction1 = (control_point_position - closest_workspace_boundary_point) / closest_distance
        term2 = - (1/2) * KR * (((1/closest_distance) - (1/threshold))**2) * n * (distance_to_cp_goal**(n-1))
        direction2 = (control_point_position - cp_goal_position) / distance_to_cp_goal
        repulsive_cart_vel = term1*direction1 + term2*direction2

        # Compute joint space repulsion (convert Cartesian velocity to joint velocity)
        repulsive_joint_vel = closest_control_point.compute_velocity_joints_space(joints, repulsive_cart_vel)

        return repulsive_joint_vel, repulsive_cart_vel, control_point_position, closest_workspace_boundary_point   