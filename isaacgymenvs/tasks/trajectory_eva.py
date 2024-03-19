import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def trajectory_eva_1(data):
        
    # 提取x、y、z坐标
    x_coords = data.iloc[:, 0]
    y_coords = data.iloc[:, 1]
    z_coords = data.iloc[:, 2]
    num_elements = len(z_coords)
    # print("first x: ", data.iloc[:, 0][0])

    # 计算数据范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    # print("y_min: ", y_min)
    # print("y_max: ", y_max)
    # print("z_min: ", z_min)
    # print("z_max: ", z_max)
    # 计算立方体尺寸
    cube_size = 0.1

    # 计算立方体的边界
    x_edges = np.arange(x_min, x_max + cube_size, cube_size)
    y_edges = np.arange(y_min, y_max + cube_size, cube_size)
    z_edges = np.arange(z_min, z_max + cube_size, cube_size)
    # print("x_edges: ", x_edges)
    # print("y_edges: ", y_edges)
    # print("z_edges: ", z_edges)
    # 计算每个小正方体中的坐标点数量
    counts = np.zeros((len(x_edges), len(y_edges), len(z_edges)), dtype=int)

    # 遍历每个坐标点，将其分配到相应的小正方体中
    for x, y, z in zip(x_coords, y_coords, z_coords):
        x_index = np.searchsorted(x_edges, x) - 1
        y_index = np.searchsorted(y_edges, y) - 1
        z_index = np.searchsorted(z_edges, z) - 1
        counts[x_index, y_index, z_index] += 1

    sum = 0
    N = 0
    sum_cube = 0
    # 输出每个小正方体里的坐标点个数
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            for k in range(counts.shape[2]):
                # print(f'小正方体({i},{j},{k})里的坐标点个数: {counts[i,j,k]}')
                N = N + counts[i,j,k]
                # sum = sum + counts[i,j,k]**2
                sum_cube = sum_cube + 1
                sum = sum + (counts[i,j,k]/(num_elements - cube_size**2))**2
    # print("sum: ", sum)
    # print("N: ", N)
    C = np.log(sum/(N**2))/np.log(cube_size)
    # C = sum
    
    return C

def trajectory_eva_2(data):
    num_elements = len(data)
    evals = []
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
        angle_deg = np.degrees(angle_rad)+90
        evals.append(angle_deg)
    C = sum(evals) / num_elements
    return C
        


line_x = pd.read_csv('isaacgymenvs/tasks/trajectory/line_x.csv')
line_xy = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xy.csv')
line_xyz = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xyz.csv')
circle = pd.read_csv('isaacgymenvs/tasks/trajectory/circle.csv')
d_circle = pd.read_csv('isaacgymenvs/tasks/trajectory/d_circle.csv')
d_circle_plus = pd.read_csv('isaacgymenvs/tasks/trajectory/d_circle_plus.csv')
helix = pd.read_csv('isaacgymenvs/tasks/trajectory/helix.csv')


# C_line_x = trajectory_eva_1(line_x)
# C_line_xy = trajectory_eva_1(line_xy)
# C_line_xyz = trajectory_eva_1(line_xyz)
# C_circle = trajectory_eva_1(circle)
# C_d_circle = trajectory_eva_1(d_circle)
# C_d_circle_plus = trajectory_eva_1(d_circle_plus)
# C_helix = trajectory_eva_1(helix)

C_line_x = 0
C_line_xy = 0
C_line_xyz = 0
C_circle = trajectory_eva_2(circle)
C_d_circle = trajectory_eva_2(d_circle)
C_d_circle_plus = trajectory_eva_2(d_circle_plus)
C_helix = trajectory_eva_2(helix)

print("C_line_x: ", C_line_x)
print("C_line_xy: ", C_line_xy)
print("C_line_xyz: ", C_line_xyz)
print("C_circle: ", C_circle)
print("C_d_circle: ", C_d_circle)
print("C_d_circle_plus: ", C_d_circle_plus)
print("C_helix: ", C_helix)




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
original_numbers.append(C_d_circle)
original_numbers.append(C_d_circle_plus)
original_numbers.append(C_helix)

scaled_values = scale_to_integer_range(original_numbers, 1, 5)


# Plotting the bar chart
plt.bar(range(len(scaled_values)), scaled_values, color=['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink'])
plt.xlabel('Trajectory')
plt.ylabel('Complexity (Scaled)')
plt.title('Complexity of Trajectories')
plt.xticks(range(len(scaled_values)), ['Line X', 'Line XY', 'Line XYZ', 'Circle', 'D Circle', 'D Circle Plus', 'Helix'], fontsize=8)
plt.show()
print("scaled_values: ", scaled_values)
