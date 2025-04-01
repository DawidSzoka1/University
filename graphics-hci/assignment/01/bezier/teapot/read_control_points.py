import numpy as np

def read_control_points_from_txt(file_path, shape):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = [list(map(float, line.split())) for line in lines if line.strip()]
    return np.array(points).reshape(shape)
