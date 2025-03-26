import pygame
from pygameTutorial.config import screen, WIDTH, HEIGHT
from pygameTutorial.util import draw_text
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

def menu_screen():
    draw_text("MENU GŁÓWNE", WIDTH // 2 - 100, HEIGHT // 4)

    # Przycisk Start
    start_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 50, 200, 50)
    pygame.draw.rect(screen, GRAY, start_button)
    draw_text("START", WIDTH // 2 - 40, HEIGHT // 2 - 35, BLACK)

    # Przycisk Wyjścia
    exit_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 20, 200, 50)
    pygame.draw.rect(screen, GRAY, exit_button)
    draw_text("WYJŚCIE", WIDTH // 2 - 50, HEIGHT // 2 + 35, BLACK)

    return start_button, exit_button