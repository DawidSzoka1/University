import numpy as np

# Funkcja wczytująca punkty kontrolne z pliku tekstowego
# file_path – ścieżka do pliku tekstowego (.txt)
# shape – oczekiwany kształt tablicy numpy, np. (-1, 16, 3)
def read_control_points_from_txt(file_path, shape):
    # Otwórz plik w trybie odczytu i wczytaj wszystkie linie do listy
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Przetwórz każdą niepustą linię: zamień ciągi na liczby zmiennoprzecinkowe
    points = [list(map(float, line.split())) for line in lines if line.strip()]

    # Zwróć tablicę numpy z przekształconym kształtem (np. jako lista patchy 4x4 punktów 3D)
    return np.array(points).reshape(shape)
