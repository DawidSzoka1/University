import numpy as np

def read_control_points_from_txt(file_path, shape):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = [list(map(float, line.split())) for line in lines if line.strip()]
    if len(points) != shape[0] * shape[1]:
        raise ValueError("Nieprawidłowa liczba punktów w pliku txt " + file_path)
    return np.array(points).reshape(shape)
