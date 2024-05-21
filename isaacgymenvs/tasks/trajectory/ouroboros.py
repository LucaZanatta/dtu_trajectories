import matplotlib.pyplot as plt
import csv
import numpy as np


def generate_ouroboros_points(num_points):
    t = np.linspace(0, 2*np.pi, num_points)
    x = np.sin(t)*2
    y = np.cos(t) * np.sin(t)*1.5
    z = np.ones_like(t)
    return x, y, z


def save_to_csv(x, y, z, filename='isaacgymenvs/tasks/trajectory/ouroboros.csv'):
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
    ax.set_zlim(0,2)
    ax.set_aspect('equal') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('ouroboros', fontsize=16)
    ax.legend()
    plt.savefig('isaacgymenvs/tasks/trajectory/figure/ouroboros.png', dpi=600)
    plt.show()


num_points = 35
x, y, z = generate_ouroboros_points(num_points)

# plot_2d_trajectory(x, y)
plot_3d_trajectory(x, y, z)

save_to_csv(x, y, z)