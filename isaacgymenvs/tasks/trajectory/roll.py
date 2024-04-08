import matplotlib.pyplot as plt
import csv
import numpy as np


def generate_roll_points():
    t1 = np.linspace(0, 0.4, 4)
    x1 = t1
    y1 = np.zeros_like(x1)
    z1 = np.ones_like(x1)
    
    radius = 0.5
    t2 = np.linspace(-np.pi/2, np.pi*0.99/2, 20)
    x2 = radius*np.cos(t2)+0.5
    y2 = np.zeros_like(x2)
    z2 = radius*np.sin(t2)+1.5
    
    t3 = np.linspace(np.pi/2, np.pi*3/2, 20)
    x3 = radius*np.cos(t3)+0.5
    y3 = np.zeros_like(x3)
    z3 = radius*np.sin(t3)+1.5
    
    t4 = np.linspace(0.6, 1.5, 9)
    x4 = t4
    y4 = np.zeros_like(x4)
    z4 = np.ones_like(x4)
    
    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    z = np.concatenate([z1, z2, z3, z4])
    
    return x, y, z


def save_to_csv(x, y, z, filename='isaacgymenvs/tasks/trajectory/roll.csv'):
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
    ax.set_xlim(0,2)
    ax.set_ylim(0,1)
    ax.set_zlim(0,2)    
    ax.set_aspect('equal') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('roll', fontsize=16)
    ax.legend()
    plt.savefig('isaacgymenvs/tasks/trajectory/figure/roll.png', dpi=600)
    plt.show()

x, y, z = generate_roll_points()

plot_2d_trajectory(x, y)
plot_3d_trajectory(x, y, z)

save_to_csv(x, y, z)



