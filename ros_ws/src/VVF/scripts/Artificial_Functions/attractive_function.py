import numpy as np

def attractive_cartesian(goal_position, tool_position, time_elapsed, change_attr_point=0.2, KP=0.35):
    cartesian_velocities = np.zeros(3)
    acceleration_max = 0.1 # m/s^2

    vector_to_goal = goal_position - tool_position
    distance_to_goal = np.linalg.norm(goal_position - tool_position)
    direction = vector_to_goal / distance_to_goal
    
    # Define a smooth acceleration function (sigmoid-based ramping)
    l = acceleration_max/(change_attr_point * KP)
    acceleration_smoothing = 1 - np.exp(-l * time_elapsed)  # Smooth transition factor (0 → 1 over time)

    if distance_to_goal <= change_attr_point:
        cartesian_velocities = KP * vector_to_goal
    else:
        cartesian_velocities =  acceleration_smoothing * change_attr_point * KP * direction
        
        

    return cartesian_velocities
