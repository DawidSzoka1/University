import numpy as np
import pyvista as pv
from read_control_points import read_control_points_from_txt
from scipy.special import comb

# Ten kod wykorzystuje biblioteki NumPy, PyVista oraz SciPy
# do tworzenia i renderowania 3D obiektów (czajnik, łyżka, filiżanka)
# na podstawie powierzchni Béziera zdefiniowanych przez punkty kontrolne.


# Funkcja Bernsteina - oblicza wartość funkcji Bernsteina dla danych parametrów
def bernstein(n, i, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


# Funkcja generująca siatkę 3D (X, Y, Z) na podstawie 16 punktów kontrolnych powierzchni Béziera
def bezier_patch(control_points, resolution=100):
    # Tworzenie siatki parametrów u i w (dwuwymiarowa siatka)
    u_vals = np.linspace(0, 1, resolution)
    w_vals = np.linspace(0, 1, resolution)
    u_mesh, w_mesh = np.meshgrid(u_vals, w_vals)

    # Inicjalizacja macierzy na współrzędne X, Y, Z
    px = np.zeros_like(u_mesh, dtype=float)
    py = np.zeros_like(u_mesh, dtype=float)
    pz = np.zeros_like(u_mesh, dtype=float)

    # Obliczanie punktów powierzchni Béziera na podstawie funkcji Bernsteina
    for i in range(4):
        for j in range(4):
            b_u = bernstein(3, i, u_mesh)
            b_w = bernstein(3, j, w_mesh)
            index = i * 4 + j
            px += control_points[index, 0] * b_u * b_w
            py += control_points[index, 1] * b_u * b_w
            pz += control_points[index, 2] * b_u * b_w

    return px, py, pz


# Funkcja tworząca siatkę (mesh) z punktów X, Y, Z do wyświetlenia w PyVista
def create_mesh(px, py, pz):
    # Łączenie współrzędnych w jeden zbiór punktów
    points = np.column_stack((px.ravel(), py.ravel(), pz.ravel()))
    faces = []
    resolution = px.shape[0]

    # Tworzenie czworokątnych ścian (faces) siatki
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = i * resolution + (j + 1)
            idx3 = (i + 1) * resolution + (j + 1)
            idx4 = (i + 1) * resolution + j
            faces.append([4, idx1, idx2, idx3, idx4])  # '4' oznacza czworokąt

    # Tworzenie obiektu PolyData z punktów i ścian
    mesh = pv.PolyData(points, np.hstack(faces))
    return mesh


# Funkcja wyświetlająca obiekty (czajnik, łyżkę i filiżankę) w oknie 3D
def plot_objects(teapot_cp, spoon_cp, cup_cp):
    plotter = pv.Plotter()

    # Pozycje przesunięcia obiektów w przestrzeni
    offsets = [(0, 0, 0), (3, 0, 0), (-4, 0, 0)]
    colors = ['blue', 'red', 'green']
    objects = [teapot_cp, spoon_cp, cup_cp]

    # Iteracja po każdym obiekcie
    for obj, offset, color in zip(objects, offsets, colors):
        for patch in obj:
            # Generowanie powierzchni Béziera dla danego patcha
            px, py, pz = bezier_patch(patch)

            # Przesunięcie pozycji w przestrzeni 3D
            px += offset[0]
            py += offset[1]
            pz += offset[2]

            # Dla filiżanki (zielona), obrót osi Y i Z (transformacja)
            if color == 'green':
                py, pz = -pz, py

            # Tworzenie siatki i dodanie jej do plottera
            mesh = create_mesh(px, py, pz)
            plotter.add_mesh(mesh, color=color, opacity=0.6)

    # Wyświetlenie okna renderowania
    plotter.show()


# Wczytywanie punktów kontrolnych z plików tekstowych
# Każdy plik zawiera zestawy punktów 3D (16 na patch) w formacie: (patch_count, 16, 3)
teapot_cp = read_control_points_from_txt("teapotCGA.bpt.txt", (-1, 16, 3))
spoon_cp = read_control_points_from_txt("teaspoon.bpt.txt", (-1, 16, 3))
cup_cp = read_control_points_from_txt("teacup.bpt.txt", (-1, 16, 3))

# Wywołanie funkcji renderującej wszystkie obiekty 3D
plot_objects(teapot_cp, spoon_cp, cup_cp)
