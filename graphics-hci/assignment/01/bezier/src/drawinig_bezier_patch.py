import numpy as np
import pyvista as pv
from read_control_points import read_control_points_from_txt
from scipy.special import comb


def bernstein(n, i, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_patch(control_points, resolution=10):
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
    return px, py, pz


def create_mesh(px, py, pz, color):
    points = np.column_stack((px.ravel(), py.ravel(), pz.ravel()))
    faces = []
    resolution = px.shape[0]
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = i * resolution + (j + 1)
            idx3 = (i + 1) * resolution + (j + 1)
            idx4 = (i + 1) * resolution + j
            faces.append([4, idx1, idx2, idx3, idx4])
    mesh = pv.PolyData(points, np.hstack(faces))
    return mesh


def plot_objects(teapot_cp, spoon_cp, cup_cp):
    plotter = pv.Plotter()
    offsets = [(0, 0, 0), (3, 0, 0), (-4, 0, 0)]
    colors = ['blue', 'red', 'green']
    objects = [teapot_cp, spoon_cp, cup_cp]

    for obj, offset, color in zip(objects, offsets, colors):
        for patch in obj:
            px, py, pz = bezier_patch(patch)
            px += offset[0]
            py += offset[1]
            pz += offset[2]

            if color == 'green':
                py, pz = -pz, py

            mesh = create_mesh(px, py, pz, color)
            plotter.add_mesh(mesh, color=color, opacity=0.6)

    plotter.show()


teapot_cp = read_control_points_from_txt("teapotCGA.bpt.txt", (-1, 16, 3))
spoon_cp = read_control_points_from_txt("teaspoon.bpt.txt", (-1, 16, 3))
cup_cp = read_control_points_from_txt("teacup.bpt.txt", (-1, 16, 3))
plot_objects(teapot_cp, spoon_cp, cup_cp)
