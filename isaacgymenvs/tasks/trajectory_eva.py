import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def trajectory_eva_1(data):
        
    x_coords = data.iloc[:, 0]
    y_coords = data.iloc[:, 1]
    z_coords = data.iloc[:, 2]
    num_elements = len(z_coords)

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    cube_size = 0.1

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
                # sum_cube = sum_cube + 1
                # sum = sum + (counts[i,j,k]/(num_elements - cube_size**2))**2
    # print("sum: ", sum)
    # print("N: ", N)
    C = np.log(sum / (N**2)) / np.log(cube_size)
    # C = np.log(sum/(N))/np.log(cube_size)
    # C = np.log(sum)/np.log(cube_size)
    # C = sum
    return C

def trajectory_eva_2(data):

    x_coords = data.iloc[:, 0]
    y_coords = data.iloc[:, 1]
    z_coords = data.iloc[:, 2]
    num_elements = len(z_coords)

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    cube_size = 0.1

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
                # sum = sum + counts[i,j,k]**2
                sum_cube = sum_cube + 1
                sum = sum + (counts[i,j,k]/(num_elements - cube_size**2))**2
    # print("sum: ", sum)
    # print("N: ", N)
    # C = np.log(sum / (N**2)) / np.log(cube_size)
    C = np.log(sum/(N))/np.log(cube_size)
    # C = np.log(sum)/np.log(cube_size)
    # C = sum
    return C


def trajectory_eva_3(data):
    # print("data: ")
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
        cos_angle = dot_product / (norm_vector1 * norm_vector2)
        angle_clip = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(angle_clip)
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)

    # sun of lengths of the trajectory
    for i in range(num_elements-1):
        point1 = data.loc[i, ['X', 'Y', 'Z']]
        point2 = data.loc[i+1, ['X', 'Y', 'Z']]
        vector = point2 - point1
        norm_vector = np.linalg.norm(vector)
        lengths.append(norm_vector)
    
    # print("angles: ", sum(angles))
    # print("lengths: ", sum(lengths))
    C = sum(angles)/np.log(sum(lengths))
    return C

        


line_x = pd.read_csv('isaacgymenvs/tasks/trajectory/line_x.csv')
line_xy = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xy.csv')
line_xyz = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xyz.csv')
circle = pd.read_csv('isaacgymenvs/tasks/trajectory/circle.csv')
ouroboros = pd.read_csv('isaacgymenvs/tasks/trajectory/ouroboros.csv')
ouroboros_z = pd.read_csv('isaacgymenvs/tasks/trajectory/ouroboros_z.csv')
ouroboros_plus = pd.read_csv('isaacgymenvs/tasks/trajectory/ouroboros_plus.csv')
ouroboros_plus_z = pd.read_csv('isaacgymenvs/tasks/trajectory/ouroboros_plus_z.csv')
corss = pd.read_csv('isaacgymenvs/tasks/trajectory/corss.csv')
ellipse = pd.read_csv('isaacgymenvs/tasks/trajectory/ellipse.csv')
line_fold = pd.read_csv('isaacgymenvs/tasks/trajectory/line_fold.csv')
playground = pd.read_csv('isaacgymenvs/tasks/trajectory/playground.csv')
retreat = pd.read_csv('isaacgymenvs/tasks/trajectory/retreat.csv')
sin = pd.read_csv('isaacgymenvs/tasks/trajectory/sin.csv')
spiral_v = pd.read_csv('isaacgymenvs/tasks/trajectory/spiral_v.csv')
spiral_h = pd.read_csv('isaacgymenvs/tasks/trajectory/spiral_h.csv')
swirl = pd.read_csv('isaacgymenvs/tasks/trajectory/swirl.csv')
tornado = pd.read_csv('isaacgymenvs/tasks/trajectory/tornado.csv')
wheel = pd.read_csv('isaacgymenvs/tasks/trajectory/wheel.csv')
roll = pd.read_csv('isaacgymenvs/tasks/trajectory/roll.csv')

# m1
# m1_line_x = trajectory_eva_1(line_x)
# m1_line_xy = trajectory_eva_1(line_xy)
# m1_line_xyz = trajectory_eva_1(line_xyz)
# m1_circle = trajectory_eva_1(circle)
# m1_ouroboros = trajectory_eva_1(ouroboros)
# m1_ouroboros_z = trajectory_eva_1(ouroboros_z)
# m1_ouroboros_plus = trajectory_eva_1(ouroboros_plus)
# m1_ouroboros_plus_z = trajectory_eva_1(ouroboros_plus_z)
# m1_corss = trajectory_eva_1(corss)
# m1_ellipse = trajectory_eva_1(ellipse)
# m1_line_fold = trajectory_eva_1(line_fold)
# m1_playground = trajectory_eva_1(playground)
# m1_retreat = trajectory_eva_1(retreat)
# m1_sin = trajectory_eva_1(sin)
# m1_spiral_v = trajectory_eva_1(spiral_v)
# m1_spiral_h = trajectory_eva_1(spiral_h)
# m1_swirl = trajectory_eva_1(swirl)
# m1_tornado = trajectory_eva_1(tornado)
# m1_wheel = trajectory_eva_1(wheel)
# m1_roll = trajectory_eva_1(roll)

# m2
# m2_line_x = trajectory_eva_2(line_x)
# m2_line_xy = trajectory_eva_2(line_xy)
# m2_line_xyz = trajectory_eva_2(line_xyz)
# m2_circle = trajectory_eva_2(circle)
# m2_ouroboros = trajectory_eva_2(ouroboros)
# m2_ouroboros_z = trajectory_eva_2(ouroboros_z)
# m2_ouroboros_plus = trajectory_eva_2(ouroboros_plus)
# m2_ouroboros_plus_z = trajectory_eva_2(ouroboros_plus_z)
# m2_corss = trajectory_eva_2(corss)
# m2_ellipse = trajectory_eva_2(ellipse)
# m2_line_fold = trajectory_eva_2(line_fold)
# m2_playground = trajectory_eva_2(playground)
# m2_retreat = trajectory_eva_2(retreat)
# m2_sin = trajectory_eva_2(sin)
# m2_spiral_v = trajectory_eva_2(spiral_v)
# m2_spiral_h = trajectory_eva_2(spiral_h)
# m2_swirl = trajectory_eva_2(swirl)
# m2_tornado = trajectory_eva_2(tornado)
# m2_wheel = trajectory_eva_2(wheel)
# m2_roll = trajectory_eva_2(roll)

# m3
m3_line_x = 0
m3_line_xy = 0
m3_line_xyz = 0
m3_circle = trajectory_eva_3(circle)
m3_ouroboros = trajectory_eva_3(ouroboros)
m3_ouroboros_z = trajectory_eva_3(ouroboros_z)
m3_ouroboros_plus = trajectory_eva_3(ouroboros_plus)
m3_ouroboros_plus_z = trajectory_eva_3(ouroboros_plus_z)
m3_corss = trajectory_eva_3(corss)
m3_ellipse = trajectory_eva_3(ellipse)
m3_playground = trajectory_eva_3(playground)
m3_retreat = trajectory_eva_3(retreat)
m3_sin = trajectory_eva_3(sin)
m3_spiral_v = trajectory_eva_3(spiral_v)
m3_spiral_h = trajectory_eva_3(spiral_h)
m3_swirl = trajectory_eva_3(swirl)
m3_tornado = trajectory_eva_3(tornado)
m3_wheel = trajectory_eva_3(wheel)
m3_roll = trajectory_eva_3(roll)


# method 1
# original_numbers = []
# original_numbers.append(m1_line_x)
# original_numbers.append(m1_line_xy)
# original_numbers.append(m1_line_xyz)
# original_numbers.append(m1_circle)
# original_numbers.append(m1_ouroboros)
# original_numbers.append(m1_ouroboros_z)
# original_numbers.append(m1_ouroboros_plus)
# original_numbers.append(m1_ouroboros_plus_z)
# original_numbers.append(m1_corss)
# original_numbers.append(m1_ellipse)
# original_numbers.append(m1_line_fold)
# original_numbers.append(m1_playground)
# original_numbers.append(m1_retreat)
# original_numbers.append(m1_sin)
# original_numbers.append(m1_spiral_v)
# original_numbers.append(m1_spiral_h)
# original_numbers.append(m1_swirl)
# original_numbers.append(m1_tornado)
# original_numbers.append(m1_wheel)
# original_numbers.append(m1_roll)

# method 2
# original_numbers = []
# original_numbers.append(m2_line_x)
# original_numbers.append(m2_line_xy)
# original_numbers.append(m2_line_xyz)
# original_numbers.append(m2_circle)
# original_numbers.append(m2_ouroboros)
# original_numbers.append(m2_ouroboros_z)
# original_numbers.append(m2_ouroboros_plus)
# original_numbers.append(m2_ouroboros_plus_z)
# original_numbers.append(m2_corss)
# original_numbers.append(m2_ellipse)
# original_numbers.append(m2_line_fold)
# original_numbers.append(m2_playground)
# original_numbers.append(m2_retreat)
# original_numbers.append(m2_sin)
# original_numbers.append(m2_spiral_v)
# original_numbers.append(m2_spiral_h)
# original_numbers.append(m2_swirl)
# original_numbers.append(m2_tornado)
# original_numbers.append(m2_wheel)
# original_numbers.append(m2_roll)

# method 3
original_numbers = []
original_numbers.append(m3_line_x)
original_numbers.append(m3_line_xy)
original_numbers.append(m3_line_xyz)
original_numbers.append(m3_circle)
original_numbers.append(m3_ouroboros)
original_numbers.append(m3_ouroboros_z)
original_numbers.append(m3_ouroboros_plus)
original_numbers.append(m3_ouroboros_plus_z)
original_numbers.append(m3_corss)
original_numbers.append(m3_ellipse)
original_numbers.append(m3_playground)
original_numbers.append(m3_retreat)
original_numbers.append(m3_sin)
original_numbers.append(m3_spiral_v)
original_numbers.append(m3_spiral_h)
original_numbers.append(m3_swirl)
original_numbers.append(m3_tornado)
original_numbers.append(m3_wheel)
original_numbers.append(m3_roll)
# print("original_numbers: ", original_numbers)

def scale_to_integer_range(values, new_min, new_max):
    min_original = min(values)
    max_original = max(values)
    scaled_values = [
        int((value - min_original) * (new_max - new_min) / (max_original - min_original) + new_min)
        for value in values
    ]
    return scaled_values

scaled_values = scale_to_integer_range(original_numbers, 0, 10)


# Create a colormap
colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'black': (0, 0, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'brown': (165, 42, 42),
    'teal': (0, 128, 128),
    'lime': (0, 255, 0),
    'olive': (128, 128, 0),
    'maroon': (128, 0, 0),
    'navy': (0, 0, 128),
    'aquamarine': (127, 255, 212),
    'turquoise': (64, 224, 208),
    'gold': (255, 215, 0),
    'silver': (192, 192, 192),
    'indigo': (75, 0, 130),
    'violet': (238, 130, 238),
    'azure': (240, 255, 255),
    'lavender': (230, 230, 250),
    'beige': (245, 245, 220),
    'khaki': (240, 230, 140),
    'salmon': (250, 128, 114),
    'coral': (255, 127, 80),
    'tan': (210, 180, 140)
}

# Plotting the bar chart
plt.bar(range(len(scaled_values)), scaled_values, color=colors)
plt.xlabel('Trajectory', fontsize=12)
plt.ylabel('Complexity (Scaled)',fontsize=12)
plt.title('Complexity of Trajectories')
plt.xticks(range(len(scaled_values)), ['Line X', 'Line XY', 'Line XYZ', 'Circle', 'Ouroboros', 'Ouroboros_z', 'Ouroboros_plus', 'Ouroboros_plus_z', 'Cross', 'Ellipse', 'Playground', 'Retreat', 'Sin', 'Spiral V', 'Spiral H', 'Swirl', 'Tornado', 'Wheel', 'Roll'], fontsize=10, rotation=30)
plt.show()

# print("scaled_values: ", scaled_values)
