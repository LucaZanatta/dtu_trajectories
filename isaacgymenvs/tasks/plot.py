import pandas as pd

import matplotlib.pyplot as plt

# # Read the data from velocity_log.csv
# velocity_log = pd.read_csv('log/velocity_log.csv')

# # Extract the x and y values from the data
# x = velocity_log.index
# y = velocity_log

# # Plot the line chart
# plt.plot(x, y)
# plt.xlabel('Steps')
# plt.ylabel('Velocity')
# plt.title('Velocity over Steps')
# plt.show()

# # Read the data from body_drone_linvels_log.csv
# body_drone_linvels_log = pd.read_csv('log/body_drone_linvels_log.csv')
# # Extract the x and y values from the new data
# body_drone_linvels_log_x = body_drone_linvels_log.index
# body_drone_linvels_log_y_1 = body_drone_linvels_log.iloc[:, 0]
# body_drone_linvels_log_y_2 = body_drone_linvels_log.iloc[:, 1]
# body_drone_linvels_log_y_3 = body_drone_linvels_log.iloc[:, 2]

# plt.plot(body_drone_linvels_log_x, body_drone_linvels_log_y_1, label='velocity on x', color='r')
# plt.plot(body_drone_linvels_log_x, body_drone_linvels_log_y_2, label='velocity on y', color='g')
# plt.plot(body_drone_linvels_log_x, body_drone_linvels_log_y_3, label='velocity on z', color='b')
# plt.xlabel('Steps')
# plt.ylabel('Velocity')
# plt.title('body_drone_linvels on x y z axis over Steps')
# plt.legend()
# plt.show()

# Read the data from body_drone_linvels_log.csv
forces_log = pd.read_csv('log/forces_log.csv')
force_log_x = forces_log.index
# force_log_y_1 = forces_log.iloc[:, 0]
# force_log_y_2 = forces_log.iloc[:, 1]
force_log_y_3 = forces_log.iloc[:, 2]

# plt.plot(force_log_x, force_log_y_1, label='force on x', color='r')
# plt.plot(force_log_x, force_log_y_2, label='force on y', color='g')
plt.plot(force_log_x, force_log_y_3, label='force on z', color='b')
plt.xlabel('Steps')
plt.ylabel('Force')
plt.title('force on x y z axis over Steps')
plt.legend()
plt.show()

# # friction_log
# friction_log = pd.read_csv('log/friction_log.csv')
# friction_x = friction_log.index
# friction_1 = friction_log.iloc[:, 0]
# friction_2 = friction_log.iloc[:, 1]
# friction_3 = friction_log.iloc[:, 2]

# plt.plot(friction_x, friction_1, label='friction on x', color='r')
# plt.plot(friction_x, friction_2, label='friction on y', color='g')
# plt.plot(friction_x, friction_3, label='friction on z', color='b')
# plt.xlabel('Steps')
# plt.ylabel('friction')
# plt.title('friction on x y z axis over Steps')
# plt.legend()
# plt.show()

# # root_linvels_log
# root_linvels_log = pd.read_csv('log/root_linvels_log.csv')
# root_linvels_x = root_linvels_log.index
# root_linvels_y_1 = root_linvels_log.iloc[:, 0]
# root_linvels_y_2 = root_linvels_log.iloc[:, 1]
# root_linvels_y_3 = root_linvels_log.iloc[:, 2]

# plt.plot(root_linvels_x, root_linvels_y_1, label='root_linvels on x', color='r')
# plt.plot(root_linvels_x, root_linvels_y_2, label='root_linvels on y', color='g')
# plt.plot(root_linvels_x, root_linvels_y_3, label='root_linvels on z', color='b')
# plt.xlabel('Steps')
# plt.ylabel('root_linvels')
# plt.title('root_linvels on x y z axis over Steps')
# plt.legend()
# plt.show()

# reward_log = pd.read_csv('log/reward.csv')
# reward_index = reward_log.index
# reward = reward_log.iloc[:, 0]

# plt.plot(reward_index, reward, label='reward', color='r')
# plt.xlabel('Steps')
# plt.ylabel('reward')
# plt.title('reward over Steps')
# plt.legend()
# plt.show()

# target_dist_log = pd.read_csv('log/target_dist.csv')
# target_dist_log_index = target_dist_log.index
# target_dist = target_dist_log.iloc[:, 0]

# plt.plot(target_dist_log_index, target_dist, label='target_dist', color='r')
# plt.xlabel('Steps')
# plt.ylabel('target_dist')
# plt.title('target_dist over Steps')
# plt.legend()
# plt.show()


# Read the data from body_drone_linvels_log.csv
torque_log = pd.read_csv('log/torque_log.csv')
torque_log_index = torque_log.index
torque_log_x = torque_log.iloc[:, 0]
torque_log_y = torque_log.iloc[:, 1]
torque_log_z = torque_log.iloc[:, 2]

plt.plot(torque_log_index, torque_log_x, label='torque on x', color='r')
plt.plot(torque_log_index, torque_log_y, label='torque on y', color='g')
plt.plot(torque_log_index, torque_log_z, label='torque on z', color='b')
plt.xlabel('Steps')
plt.ylabel('torque')
plt.title('torque on x y z axis over Steps')
plt.legend()
plt.show()