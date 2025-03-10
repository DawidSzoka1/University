import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import comb

matplotlib.use('TkAgg')
plt.ion()

def bernstein(n, i, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_patch(control_points, resolution=20):
    u_vals = np.linspace(0, 1, resolution)
    w_vals = np.linspace(0, 1, resolution)
    u_mesh, w_mesh = np.meshgrid(u_vals, w_vals)
    px = np.zeros_like(u_mesh, dtype=float)
    py = np.zeros_like(u_mesh, dtype=float)
    pz = np.zeros_like(u_mesh, dtype=float)

    for i in range(4):
        for j in range(4):
            b_u = bernstein(3, i, u_mesh)
            b_w = bernstein(3, j, w_mesh)
            index = i * 4 + j
            px += control_points[index, 0] * b_u * b_w
            py += control_points[index, 1] * b_u * b_w
            pz += control_points[index, 2] * b_u * b_w

    return px, py, pz  # Zwracamy osobne macierze zamiast jednej tablicy


def read_control_points(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    control_points = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        control_points.append(values)
    return np.array(control_points).reshape(-1, 16, 3)


def plot_objects(teapot_cp, spoon_cp, cup_cp):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Offsets to position objects separately
    teapot_offset = np.array([0, 0, 0])
    spoon_offset = np.array([3, 0, 0])  # Move spoon to the right
    cup_offset = np.array([-3, 0, 0])  # Move cup to the left

    # Render teapot
    for k in range(teapot_cp.shape[0]):
        px, py, pz = bezier_patch(teapot_cp[k])
        ax.plot_surface(px + teapot_offset[0],
                        py + teapot_offset[1],
                        pz + teapot_offset[2], color='b', alpha=0.6)

    # Render spoon
    for k in range(spoon_cp.shape[0]):
        px, py, pz = bezier_patch(spoon_cp[k])
        ax.plot_surface(px + spoon_offset[0],
                        py + spoon_offset[1],
                        pz + spoon_offset[2], color='r', alpha=0.6)

    # Render cup
    for k in range(cup_cp.shape[0]):
        px, py, pz = bezier_patch(cup_cp[k])
        ax.plot_surface(px + cup_offset[0],
                        py + cup_offset[1],
                        pz + cup_offset[2], color='g', alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    plt.show(block=True)


teapot_cp = read_control_points("teapotCGA.bpt.txt")
spoon_cp = read_control_points("teaspoon.bpt.txt")
cup_cp = read_control_points("teacup.bpt.txt")
plot_objects(teapot_cp, spoon_cp, cup_cp)