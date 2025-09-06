import numpy as np
import rospy

def dynamic_repulsive(dynamic_obstacle, control_points):
    closest_control_point = None
    closest_obstacle_point = None
    cp_vel_final = None
    collision_point_final = np.zeros(3)
    closest_distance = float('inf')
    min_collision_time = 0.0

    
    for control_point in control_points:
            
        control_point_position = control_point.position
        
        # We compute distances from control point to all obstacle points
        distances = np.linalg.norm(dynamic_obstacle.points - control_point_position, axis=1)
        
        # We get the index of the closest obstacle point
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        cp_vel = np.zeros(3)
        posible_collision_bool, collision_point, collision_time = possible_collision(dynamic_obstacle.points[min_index], dynamic_obstacle.velocity, control_point_position, cp_vel) 
        # We update the closest values if a smaller distance is found
        if (min_distance < closest_distance) and posible_collision_bool:
            closest_distance = min_distance
            closest_control_point = control_point
            closest_obstacle_point = dynamic_obstacle.points[min_index]
            cp_vel_final = cp_vel
            collision_point_final = collision_point
            min_collision_time = collision_time
            
    if closest_control_point != None:
        rospy.loginfo(f" closest cp with collision: {closest_control_point.name} with distance {min_distance} and time {min_collision_time}")
    else:
        rospy.loginfo("no collision")
    
    
    return closest_control_point, closest_obstacle_point, closest_distance, cp_vel_final, collision_point_final

#cylinder l=0.4 beta=2.5
def compute_dynamic_repulsive(control_point_pos, obstacle_point_pos, control_point_vel, obstacle_point_vel, distance, lambda_const=0.4, beta=2.5, max_speed=0.5):#0.4
    cart_vel = np.zeros(3)
    
    # relative position and relative velocity of the control point with respect to the obstacle(closest point)
    x_rel = control_point_pos - obstacle_point_pos
    v_rel = control_point_vel - obstacle_point_vel
    
    # Same magnidute and direction (same motion) 
    if np.linalg.norm(v_rel) == 0:
        return cart_vel
    
    # We compute the cosine of the angle θ between the relative velocity (v_rel) and position vector (x_rel)
    cos_theta = np.dot(v_rel, x_rel) / (np.linalg.norm(v_rel) * distance)
    
    # Ensure numerical stability by clipping the value between -1 and 1 (to avoid floating point errors)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta = np.arccos(cos_theta)

    # **Check if θ is in the valid range for repulsive force activation**
    # The repulsive potential is **active** only when:  π/2 < θ ≤ π
    #
    # **θ > π/2**: The end-effector is **moving toward** the obstacle, increasing collision risk.
    # **θ ≤ π/2**: The end-effector is **moving away**, so no repulsion is needed.
    # **Geometric Interpretation of θ:**
    # **θ = 0** → Moving **directly away** 
    # **θ = π/2** → Moving **perpendicular** to the obstacle 
    # **θ = π** → Moving **directly toward** the obstacle → **Maximum repulsion applied**.
    if not (((np.pi/2)+0.55) < theta <= (np.pi)): #+0.68
        rospy.loginfo(f" no valid theta {theta} return 0 speed")
        return cart_vel 
    
    grad_d_x_rel = x_rel / distance
    grad_cos_theta = (v_rel / (np.linalg.norm(v_rel) * distance)) - (cos_theta / distance) * grad_d_x_rel # Gradient of cos(theta)

    factor = lambda_const * (-cos_theta) ** (beta - 1) * (np.linalg.norm(v_rel) / distance)
    grad_U_dyn = factor * ((-beta * grad_cos_theta) + (cos_theta / distance) * grad_d_x_rel)

    # **Cap the magnitude at max_speed**
    grad_norm = np.linalg.norm(grad_U_dyn)
    if grad_norm > max_speed:
        grad_U_dyn = (grad_U_dyn / grad_norm) * max_speed  # Normalize and scale
    
    #rospy.loginfo(f" dynamic repulsive: {np.linalg.norm(grad_U_dyn)}")
    #rospy.loginfo(f" theta repulsive: {theta}")


    return -grad_U_dyn  # Negative gradient of the DynamicPotential (Repulsive Cartesian Velocity)


def possible_collision(obstacle_point, obstacle_vel, cp_pos, cp_vel, max_distance=2.5, max_time=10):
    delta_c = obstacle_point - cp_pos  # Relative position
    delta_v = obstacle_vel - cp_vel # Relative velocity
    radius = 0.2#0.3

    # Solve the quadratic equation for collision time
    a = np.dot(delta_v, delta_v)
    b = 2 * np.dot(delta_c, delta_v)
    c = np.dot(delta_c, delta_c) - radius**2

    if a == 0 and c > 0:
        return False, None, 0.0

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return False, None, 0.0

    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

    #print(f"collision times: {t1}, {t2}")
    
    collision_time = None
    if t1 >= 0  and t2 >= 0:
        collision_time = min(t1, t2)
    elif t1 >= 0 and t2 < 0:
        collision_time = t1
    elif t2 >= 0 and t1 < 0:
        collision_time = t2

    if collision_time is None or collision_time > max_time:
        return False, None, 0.0

    collision_point_r = cp_pos + collision_time * cp_vel

    collision_point_o = obstacle_point + collision_time * obstacle_vel

    # Check if control point moves beyond max_distance
    if np.linalg.norm(collision_point_r - cp_pos) > max_distance:
        print("Control point exceeds maximum allowed distance. No collision.")
        return False, None, 0.0

    # Compute contact point on the surface of the obstacle sphere
    direction = collision_point_r - collision_point_o
    direction /= np.linalg.norm(direction)
    collision_point = collision_point_o + direction * (radius/2)

    return True, collision_point, collision_time

    
