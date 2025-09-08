import sys
import math
import numpy as np

# Global constants
d1 = 0.057
d2 = 0.48
d3 = 0.28
d5 = 0.4273
d4 = 0.01
d6 = 0.12

# Range for joints
q1_range = 2.68
q2_min = -2.61
q2_max = 2.61
q3_min = -2.61
q3_max = 2.61

tolerance = 1e-3

def main():
    XE_target = float(sys.argv[1])
    YE_target = float(sys.argv[2])
    ZE_target = float(sys.argv[3])

    matching_combinations = []

    print(f"Xe = {XE_target:.4f}, Ye = {YE_target:.4f} , Ze = {ZE_target:.4f}")
    temp = (XE_target - 0.12) ** 2 + YE_target ** 2
    if temp < 0.001:
        print("Invalid input: XE and YE out of bounds")
        return

    # Step 1: Find q1 solutions
    q1_solutions = inverse_for_q1(XE_target, YE_target)
    q1_solutions = filter_valid_angles(q1_solutions, -q1_range, q1_range)
    #print(f"q1 solutions: {q1_solutions}")

    for q1 in q1_solutions:
        # Step 2: For each q1, find corresponding q23 solutions
        q23_solutions = inverse_for_q23(q1 ,XE_target ,YE_target, ZE_target)
        if len(q23_solutions) == 0:
            continue
        #print(f"q23 solutions for q1 = {q1:.4f}: {q23_solutions}")

        for q23 in q23_solutions:
            # Step 3: For each q23, find corresponding q2 solutions
            q2_solutions = inverse_for_q2(q23, ZE_target)
            q2_solutions = filter_valid_angles(q2_solutions, q2_min, q2_max)
            #print(f"q2 solutions for q23 = {q23:.4f}: {q2_solutions}")

            for q2 in q2_solutions:
                # Step 4: For each q2, find corresponding q3 solutions
                q3_solutions = inverse_for_q3(q2, q23)
                q3_solutions = filter_valid_angles(q3_solutions, q3_min, q3_max)
                #print(f"q3 solutions for q2 = {q2:.4f}: {q3_solutions}")

                # Print full combinations for this path
                for q3 in q3_solutions:
                    # Calculate forward kinematics for this combination
                    XE = (d1 * np.cos(q2 - q3) * np.cos(q1) -
                          d2 * np.sin(q2 - q3) * np.cos(q1) -
                          d3 * np.sin(q2) * np.cos(q1) +
                          d4 * np.sin(q1) + d6)
                    YE = (d1 * np.cos(q2 - q3) * np.sin(q1) -
                          d2 * np.sin(q2 - q3) * np.sin(q1) -
                          d3 * np.sin(q2) * np.sin(q1) -
                          d4 * np.cos(q1))
                    ZE = d1 * np.sin(q2 - q3) + d2 * np.cos(q2 - q3) + d3 * np.cos(q2) + d5
                    if (abs(XE - XE_target) < tolerance and
                        abs(YE - YE_target) < tolerance and
                        abs(ZE - ZE_target) < tolerance):
                        combination = (q1, q2, q3)
                        matching_combinations.append(combination)
                        print(f"Matching combination found: q1 = {q1:.4f}, q2 = {q2:.4f}, q3 = {q3:.4f}")

def inverse_for_q1(XE, YE):
    a = XE - d6
    b = -YE
    c = -d4
    result = trigonometric_formula(a, b, c, q1_range, False)
    if not result:
        return []
    [q1_1, q1_2] = result
    return [q1_1, q1_2]

def inverse_for_q23(q1, XE, YE, ZE):
    [a, b, c] = variables_for_q23(q1, XE, YE, ZE)
    result = trigonometric_formula(a, b, c, q1_range, True)
    if not result:
        return []

    [q23_1, q23_2] = result
    return [q23_1, q23_2]

def inverse_for_q2(q23, ZE):
    q2_solutions = []
    temp = (ZE - d5 - (d2 * math.cos(q23)) - (d1 * math.sin(q23))) / d3
    if -1 <= temp <= 1:
        q2_1 = math.acos(temp)
        q2_2 = -q2_1
        q2_solutions.append(q2_1)
        q2_solutions.append(q2_2)
    else:
        print(f"Invalid q2 for q23 = {q23:.4f}")
    return q2_solutions

def inverse_for_q3(q2, q23):
    a = math.sin(q2)
    b = math.cos(q2)
    c = -math.cos(q23)
    result = trigonometric_formula(a, b, c, q1_range, False)
    if not result:
        return []

    [q3_1, q3_2] = result
    return [q3_1, q3_2]

def variables_for_q23(q1, XE, YE, ZE):
    if math.sin(q1) == 0:
        alpha = (XE - 0.12 - (0.01*math.sin(q1)) ) / math.cos(q1)
    else:
        alpha = (YE + 0.01 * math.cos(q1)) / math.sin(q1)
    beta =  ZE - 0.4273
    gama =  (-2*alpha*0.057) - (2*beta*0.48)
    delta = (2*alpha*0.48) - (2*beta*0.057)
    epsylon = ((0.48)**2) + ((0.057)**2)
    zeta = (alpha**2)+(beta**2)+epsylon-0.0784
    a = delta
    b = gama
    c = zeta
    return [a, b, c]

def trigonometric_formula(a, b, c, angle_range, no_norml):
    r = math.sqrt(a**2 + b**2)
    phi = math.atan2(b, a)
    if abs(c) > r :
        return [] # No valid solutions exist
    arcpart = math.asin(-c / r)
    q1 = math.asin(-c / r) - phi
    q2 = math.pi - math.asin(-c / r) - phi
    if no_norml:
        return [q1, q2]
    q1 = normalize_angle(q1, angle_range)
    q2 = normalize_angle(q2, angle_range)
    return [q1, q2]

def normalize_angle(angle, angle_range):
    while angle <= -angle_range - 0.1:
        angle += 2 * math.pi
    while angle > angle_range + 0.1:
        angle -= 2 * math.pi
    return angle

def filter_valid_angles(angles, min_angle, max_angle):
    return [angle for angle in angles if min_angle - 0.1 <= angle < max_angle + 0.1]

if __name__ == "__main__":
    main()
