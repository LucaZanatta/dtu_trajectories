import matplotlib.pyplot as plt
import csv
import numpy as np


def generate_spiral_h_points(num_points):
    t = np.linspace(0, 6*np.pi, num_points)
    radius = 0.5
    pitch = 0.5
    x = pitch * t / (2*np.pi)
    y = radius * np.sin(t)
    z = - radius * np.cos(t) + 1 + radius
    
    return x, y, z


def save_to_csv(x, y, z, filename='spiral_h.csv'):
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
    ax.set_ylim(-1,1)
    ax.set_aspect('equal') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('spiral h', fontsize=16)
    ax.legend()
    # plt.savefig('isaacgymenvs/tasks/trajectory/figure/spiral_h.png', dpi=600)
    plt.savefig('figure/spiral_h.png', dpi=600)
    plt.show()


num_points = 50
x, y, z = generate_spiral_h_points(num_points)

# plot_2d_trajectory(x, y)
plot_3d_trajectory(x, y, z)
save_to_csv(x, y, z)



