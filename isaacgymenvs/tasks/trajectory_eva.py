import pandas as pd

data = pd.read_csv('isaacgymenvs/tasks/trajectory/circle.csv')

def trajectory_eva(data):
    
    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    z_data = data.iloc[:, 2]
    
    
    