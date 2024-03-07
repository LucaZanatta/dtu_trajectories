import pandas as pd

data = pd.read_csv('isaacgymenvs/tasks/trajectory/circle.csv')

def trajectory_eva(data):
    
    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    z_data = data.iloc[:, 2]

    # Calculate the number of trajectory coordinates in each cube
    cube_size = 0.1
    x_range = (x_data.max() - x_data.min())
    y_range = (y_data.max() - y_data.min())
    z_range = max((z_data.max() - z_data.min()),1)
    x_cubes = pd.cut(x_data, bins=int(x_range / cube_size), labels=False)
    y_cubes = pd.cut(y_data, bins=int(y_range / cube_size), labels=False)
    z_cubes = pd.cut(z_data, bins=int(z_range / cube_size), labels=False)
    
    cube_counts = pd.concat([x_cubes, y_cubes, z_cubes], axis=1).groupby([x_cubes, y_cubes, z_cubes]).size()
    
    print(cube_counts)
    


    
    
trajectory_eva(data)