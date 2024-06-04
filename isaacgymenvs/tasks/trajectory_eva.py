import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

def trajectory_eva_3(data):
    # print("data: ")
    num_elements = len(data)
    angles = []
    lengths = []
    curvature = []
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

# method 3
original_numbers = []
original_numbers.append(m3_line_x)
original_numbers.append(m3_line_xy)
original_numbers.append(m3_line_xyz)
original_numbers.append(m3_circle)
original_numbers.append(m3_ellipse)
original_numbers.append(m3_playground)
original_numbers.append(m3_ouroboros)
original_numbers.append(m3_ouroboros_z)
original_numbers.append(m3_ouroboros_plus)
original_numbers.append(m3_ouroboros_plus_z)
original_numbers.append(m3_sin)
original_numbers.append(m3_spiral_v)
original_numbers.append(m3_tornado)
original_numbers.append(m3_swirl)
print("original_numbers: ", original_numbers)

def scale_to_integer_range(values, new_min, new_max):
    min_original = min(values)
    max_original = max(values)
    scaled_values = [
        int((value - min_original) * (new_max - new_min) / (max_original - min_original) + new_min)
        for value in values
    ]
    return scaled_values

scaled_values = scale_to_integer_range(original_numbers, 1, 10)


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
plt.xticks(range(len(scaled_values)), ['Line x', 'Line xy', 'Line xyz', 'Circle', 'Ellipse', 'Playground','Ouroboros', 'Ouroboros z', 'Ouroboros plus', 'Ouroboros plus z', 'Sin', 'Spiral v', 'Tornado','Swirl'], fontsize=10, rotation=30)
plt.show()

# print("scaled_values: ", scaled_values)
