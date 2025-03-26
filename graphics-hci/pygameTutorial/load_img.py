import pygame
import os

def load_images(path, num_images, start=0):
    """Ładuje listę obrazków do animacji, obsługując brakujące pliki"""
    images = []
    for i in range(start, num_images+start):
        file_path = f"{path}{i}.png"
        if os.path.exists(file_path):
            images.append(pygame.image.load(file_path))
        else:
            print(f"Brak pliku: {file_path}")  # Komunikat debugowy
    return images if images else [pygame.Surface((50, 50))]
