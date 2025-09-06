import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global constants
d1 = 0.057
d2 = 0.48
d3 = 0.28
d4 = 0.01
d5 = 0.4273
d6 = 0.12

q1_range = 2.68
q2_range = 2.61
q3_range = 2.61
tolerance = 1e-3

# Placeholder for solutions count
solution_count = {
    'one_solution': 0,
    'two_solutions': 0,
    'four_solutions': 0,
    'no_solution': 0
}

# Function to filter valid angles within a specified range
def filter_valid_angles(angles, min_angle, max_angle):
    return [angle for angle in angles if min_angle - 0.1 <= angle < max_angle + 0.1]

# Placeholder inverse kinematics function
def inverse_kinematics(XE, YE, ZE):
    matching_combinations = []  # Use a list to manage combinations manually
    temp = (XE - 0.12) ** 2 + YE ** 2
    if temp < 0.001:
        return matching_combinations  # Return as a list

    # Step 1: Find q1 solutions
    q1_solutions = inverse_for_q1(XE, YE)
    q1_solutions = filter_valid_angles(q1_solutions, -q1_range, q1_range)

    for q1 in q1_solutions:
        # Step 2: For each q1, find corresponding q23 solutions
        q23_solutions = inverse_for_q23(q1, XE, YE, ZE)

        for q23 in q23_solutions:
            # Step 3: For each q23, find corresponding q2 solutions
            q2_solutions = inverse_for_q2(q23, ZE)
            q2_solutions = filter_valid_angles(q2_solutions, -q2_range, q2_range)

            for q2 in q2_solutions:
                # Step 4: For each q2, find corresponding q3 solutions
                q3_solutions = inverse_for_q3(q2, q23)
                q3_solutions = filter_valid_angles(q3_solutions, -q3_range, q3_range)

                for q3 in q3_solutions:
                    XE_calc, YE_calc, ZE_calc = calculate_forward_kinematics(q1, q2, q3)
                    if (abs(XE_calc - XE) < tolerance and
                        abs(YE_calc - YE) < tolerance and
                        abs(ZE_calc - ZE) < tolerance):
                        new_combination = (q1, q2, q3)

                        # Only add if it's not within the tolerance of existing combinations
                        if not is_duplicate(matching_combinations, new_combination, 1e-2):
                            matching_combinations.append(new_combination)

    return matching_combinations  # Return the list of unique combinations


def is_duplicate(existing_combinations, new_combination, tolerance):
    """
    Check if the new combination is within the tolerance of any existing combination.
    """
    for existing in existing_combinations:
        if (abs(existing[0] - new_combination[0]) < tolerance and
            abs(existing[1] - new_combination[1]) < tolerance and
            abs(existing[2] - new_combination[2]) < tolerance):
            return True
    return False

# Function to calculate forward kinematics for given q1, q2, q3
def calculate_forward_kinematics(q1, q2, q3):
    XE = (d1 * np.cos(q2 - q3) * np.cos(q1) -
          d2 * np.sin(q2 - q3) * np.cos(q1) -
          d3 * np.sin(q2) * np.cos(q1) +
          d4 * np.sin(q1) + d6)
    YE = (d1 * np.cos(q2 - q3) * np.sin(q1) -
          d2 * np.sin(q2 - q3) * np.sin(q1) -
          d3 * np.sin(q2) * np.sin(q1) -
          d4 * np.cos(q1))
    ZE = d1 * np.sin(q2 - q3) + d2 * np.cos(q2 - q3) + d3 * np.cos(q2) + d5
    return XE, YE, ZE

# Inverse kinematics helper functions (from the previous code)
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
        q2_solutions.extend([q2_1, q2_2])
    return q2_solutions

def inverse_for_q3(q2, q23):
    a = math.sin(q2)
    b = math.cos(q2)
    c = -math.cos(q23)
    phi = math.atan2(b, a)
    q1 = math.asin(-c) - phi
    q2 = math.pi - math.asin(-c) - phi
    q1 = normalize_angle(q1, q3_range)
    q2 = normalize_angle(q2, q3_range)
    return [q1, q2]

def trigonometric_formula(a, b, c, angle_range, no_norml):
    r = math.sqrt(a**2 + b**2)
    phi = math.atan2(b, a)
    if abs(c) > r:
        return []
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

def variables_for_q23(q1, XE, YE, ZE):
    if math.sin(q1) == 0:
        alpha = (XE - d6 - (d4 * math.sin(q1))) / math.cos(q1)
    else:
        alpha = (YE + d4 * math.cos(q1)) / math.sin(q1)
    beta = ZE - d5
    gama = (-2 * alpha * d1) - (2 * beta * d2)
    delta = (2 * alpha * d2) - (2 * beta * d1)
    epsylon = (d2**2) + (d1**2)
    zeta = (alpha**2) + (beta**2) + epsylon - (d3**2)
    return [delta, gama, zeta]

# Generate and evaluate (XE, YE, ZE) positions
position_data = []
for q1 in np.linspace(-q1_range, q1_range, 25):
    for q2 in np.linspace(-q2_range, q2_range, 25):
        for q3 in np.linspace(-q3_range, q3_range, 25):
            XE, YE, ZE = calculate_forward_kinematics(q1, q2, q3)
            solutions = inverse_kinematics(XE, YE, ZE)
            num_solutions = len(solutions)

            if num_solutions == 1:
                solution_count['one_solution'] += 1
            elif num_solutions == 2:
                solution_count['two_solutions'] += 1
            elif num_solutions == 4:
                solution_count['four_solutions'] += 1
            else:
                solution_count['no_solution'] += 1

            position_data.append((XE, YE, ZE, num_solutions))



# Print statistics
# Calculate and print the percentage of each solution category
total_positions = sum(solution_count.values())
print("\nSolution statistics:")
print(f"Positions: {total_positions}")
for key, count in solution_count.items():
    percentage = (count / total_positions) * 100
    print(f"{key.replace('_', ' ').capitalize()}: {count} ({percentage:.2f}%)")
"""
# Print one example position for each category
printed_categories = set()
print("\nExample position for each category:")
for (XE, YE, ZE, num_solutions) in position_data:
    for category, target_num in [("one_solution", 1), ("two_solutions", 2),
                                 ("four_solutions", 4), ("no_solution", 0)]:
        if num_solutions == target_num and category not in printed_categories:
            print(f"Category '{category.replace('_', ' ').capitalize()}': XE = {XE:.4f}, YE = {YE:.4f}, ZE = {ZE:.4f}")
            printed_categories.add(category)
            break
"""

def plot_category(category, data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Positions with {category.replace('_', ' ').capitalize()}")
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Filter data for this category
    category_map = {
        'one_solution': 1,
        'two_solutions': 2,
        'four_solutions': 4,
        'no_solution': 0
    }
    #print(f"category: {category}")
    #print(f"position_data: {position_data}")
    filtered_data = [point for point in data if point[3] == category_map[category]]

    for point in filtered_data:
        ax.scatter(point[0], point[1], point[2], marker='x', color='b')

    plt.show()

# Plotting data for each category
for category in solution_count.keys():
    if solution_count[category] > 0:
        plot_category(category, position_data)

# Save results to CSV
with open("position_data_with_solutions.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["XE", "YE", "ZE", "Num_Solutions"])
    writer.writerows(position_data)

print("Position data with solution counts saved to 'position_data_with_solutions.csv'")
