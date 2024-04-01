import matplotlib.pyplot as plt
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def generate_spiral_v_points(num_points):
    t = np.linspace(0, 6*np.pi, num_points)
    radius = 1
    pitch = 0.5
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t / (2*np.pi) + 1
    x = x - 1
    
    return x, y, z


def save_to_csv(x, y, z, filename='spiral_v.csv'):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y', 'Z'])
        for xi, yi, zi in zip(x, y, z):
            writer.writerow([round(xi, 3), round(yi, 3), round(zi, 3)])


def plot_2d_trajectory(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, c='black', label='Drone Trajectory')
    ax.scatter(x, y, c='red', marker='*')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()
    plt.show()


def plot_3d_trajectory(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c = 'black', label='Drone Trajectory')
    ax.scatter(x, y, z, c='red', marker='*')
    ax.scatter(x[0], y[0], z[0], c='orange', marker='o', s = 60, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='blue', marker='x', s = 200, label='End')
    ax.set_zlim(0,4)
    ax.set_aspect('equal') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('spiral v', fontsize=16)
    ax.legend()
    plt.savefig('isaacgymenvs/tasks/trajectory/figure/spiral_v.png', dpi=600)
    plt.show()



num_points = 100
x, y, z = generate_spiral_v_points(num_points)

plot_2d_trajectory(x, y)
plot_3d_trajectory(x, y, z)
save_to_csv(x, y, z)

# ############
# # 提取最小和最大的x、y、z坐标值
# min_x = x.min()
# max_x = x.max()
# min_y = y.min()
# max_y = y.max()
# min_z = z.min()
# max_z = z.max()
# # print("min_x: ", min_x)
# # print("max_x: ", max_x)
# # print("min_y: ", min_y)
# # print("max_y: ", max_y)
# # print("min_z: ", min_z)
# # print("max_z: ", max_z)

# # 绘制立方体
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, color='r')
# # 立方体的八个顶点
# # vertices = [
# #     [center_x + side_length / 2, center_y + side_length / 2, center_z + side_length / 2],
# #     [center_x + side_length / 2, center_y + side_length / 2, center_z - side_length / 2],
# #     [center_x + side_length / 2, center_y - side_length / 2, center_z + side_length / 2],
# #     [center_x + side_length / 2, center_y - side_length / 2, center_z - side_length / 2],
# #     [center_x - side_length / 2, center_y + side_length / 2, center_z + side_length / 2],
# #     [center_x - side_length / 2, center_y + side_length / 2, center_z - side_length / 2],
# #     [center_x - side_length / 2, center_y - side_length / 2, center_z + side_length / 2],
# #     [center_x - side_length / 2, center_y - side_length / 2, center_z - side_length / 2]
# # ]

# vertices = [
#     [max_x, max_y, max_z],
#     [max_x, max_y, min_z],
#     [max_x, min_y, max_z],
#     [max_x, min_y, min_z],
#     [min_x, max_y, max_z],
#     [min_x, max_y, min_z],
#     [min_x, min_y, max_z],
#     [min_x, min_y, min_z]
# ]
# # 通过将这些点连接起来形成立方体的12条边
# edges = [
#     [vertices[0], vertices[1]],
#     [vertices[0], vertices[2]],
#     [vertices[0], vertices[4]],
#     [vertices[1], vertices[3]],
#     [vertices[1], vertices[5]],
#     [vertices[2], vertices[3]],
#     [vertices[2], vertices[6]],
#     [vertices[3], vertices[7]],
#     [vertices[4], vertices[5]],
#     [vertices[4], vertices[6]],
#     [vertices[5], vertices[7]],
#     [vertices[6], vertices[7]]
# ]

# # 绘制立方体的边
# for edge in edges:
#     ax.plot3D(*zip(*edge), color='b')

# # 绘制所有点


# plt.show()
# #########