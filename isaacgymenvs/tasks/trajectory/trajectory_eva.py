import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def trajectory_eva_1(data):
        
    x_coords = data.iloc[:, 0]
    y_coords = data.iloc[:, 1]
    z_coords = data.iloc[:, 2]
    num_elements = len(z_coords)

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    cube_size = 0.2

    x_edges = np.arange(x_min, x_max + cube_size, cube_size)
    y_edges = np.arange(y_min, y_max + cube_size, cube_size)
    z_edges = np.arange(z_min, z_max + cube_size, cube_size)

    counts = np.zeros((len(x_edges), len(y_edges), len(z_edges)), dtype=int)

    for x, y, z in zip(x_coords, y_coords, z_coords):
        x_index = np.searchsorted(x_edges, x) - 1
        y_index = np.searchsorted(y_edges, y) - 1
        z_index = np.searchsorted(z_edges, z) - 1
        counts[x_index, y_index, z_index] += 1

    sum = 0
    N = 0
    sum_cube = 0

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            for k in range(counts.shape[2]):

                N = N + counts[i,j,k]
                sum = sum + counts[i,j,k]**2
                sum_cube = sum_cube + 1
                # sum = sum + (counts[i,j,k]/(num_elements - cube_size**2))**2

    C = np.log(sum / (N**2)) / np.log(cube_size)
    # C = np.log(sum)/np.log(cube_size)
    # C = np.log(sum/N)/np.log(cube_size)
    # C = sum
    
    return C

def trajectory_eva_3(data):
    num_elements = len(data)
    angles = []
    lengths = []
    # sum of angles between 3 consecutive points
    for i in range(num_elements-2):
        point1 = data.loc[i, ['X', 'Y', 'Z']]
        point2 = data.loc[i+1, ['X', 'Y', 'Z']]
        point3 = data.loc[i+2, ['X', 'Y', 'Z']]
        vector1 = point2 - point1
        vector2 = point3 - point2
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        angle_rad = np.arccos(dot_product / (norm_vector1 * norm_vector2))
        angle_deg = np.degrees(angle_rad)
        eval = (norm_vector1+norm_vector2)**2 * angle_deg
        angles.append(eval)
    
    # sun of lengths of the trajectory
    for i in range(num_elements-1):
        point1 = data.loc[i, ['X', 'Y', 'Z']]
        point2 = data.loc[i+1, ['X', 'Y', 'Z']]
        vector = point2 - point1
        norm_vector = np.linalg.norm(vector)
        lengths.append(norm_vector)
    
    C = sum(angles)/sum(lengths)
    return np.log(C)

        


line_x = pd.read_csv('isaacgymenvs/tasks/trajectory/line_x.csv')
line_xy = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xy.csv')
line_xyz = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xyz.csv')
circle = pd.read_csv('isaacgymenvs/tasks/trajectory/circle.csv')
ouroboros = pd.read_csv('isaacgymenvs/tasks/trajectory/ouroboros.csv')
ouroboros_plus = pd.read_csv('isaacgymenvs/tasks/trajectory/ouroboros_plus.csv')
spiral_v = pd.read_csv('isaacgymenvs/tasks/trajectory/spiral_v.csv')
tornado = pd.read_csv('isaacgymenvs/tasks/trajectory/tornado.csv')


# C_line_x = trajectory_eva_1(line_x)
# C_line_xy = trajectory_eva_1(line_xy)
# C_line_xyz = trajectory_eva_1(line_xyz)
# C_circle = trajectory_eva_1(circle)
# C_ouroboros = trajectory_eva_1(ouroboros)
# C_ouroboros_plus = trajectory_eva_1(ouroboros_plus)
# C_spiral_v = trajectory_eva_1(spiral_v)
# C_tornado = trajectory_eva_1(tornado)

C_line_x = 0
C_line_xy = 0
C_line_xyz = 0
C_circle = trajectory_eva_3(circle)
C_ouroboros = trajectory_eva_3(ouroboros)
C_ouroboros_plus = trajectory_eva_3(ouroboros_plus)
C_spiral_v = trajectory_eva_3(spiral_v)
C_tornado = trajectory_eva_3(tornado)

print("C_line_x: ", C_line_x)
print("C_line_xy: ", C_line_xy)
print("C_line_xyz: ", C_line_xyz)
print("C_circle: ", C_circle)
print("C_ouroboros: ", C_ouroboros)
print("C_ouroboros_plus: ", C_ouroboros_plus)
print("C_spiral_v: ", C_spiral_v)
print("C_tornado: ", C_tornado)




def scale_to_integer_range(values, new_min, new_max):
    min_original = min(values)
    max_original = max(values)
    scaled_values = [
        int((value - min_original) * (new_max - new_min) / (max_original - min_original) + new_min)
        for value in values
    ]
    return scaled_values

original_numbers = []
original_numbers.append(C_line_x)
original_numbers.append(C_line_xy)
original_numbers.append(C_line_xyz)
original_numbers.append(C_circle)
original_numbers.append(C_ouroboros)
original_numbers.append(C_ouroboros_plus)
original_numbers.append(C_spiral_v)
original_numbers.append(C_tornado)

scaled_values = scale_to_integer_range(original_numbers, 1, 10)


# Plotting the bar chart
plt.bar(range(len(scaled_values)), scaled_values, color=['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'grey'])
# plt.xlabel('Trajectory', fontsize=14)
plt.ylabel('Complexity (Scaled)', fontsize=14)
plt.title('Method 3', fontsize=18)
plt.xticks(range(len(scaled_values)), ['Line X', 'Line XY', 'Line XYZ', 'Circle', 'Ouroboros', 'Ouroboros plus', 'spiral_v', 'Tornado'], fontsize=14, rotation=45, ha='right')
plt.tight_layout()
plt.show()
print("scaled_values: ", scaled_values)
