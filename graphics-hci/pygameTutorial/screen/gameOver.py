from pygameTutorial.config import screen, WIDTH, HEIGHT, WHITE
from pygameTutorial.util import draw_text
import pygame


def game_over_screen(score, message, bg):
    screen.blit(bg, (0, 0))
    draw_text(message, WIDTH // 2 - 120, HEIGHT // 2 - 50, (255, 0, 0))
    draw_text(f"Score: {score}", WIDTH // 2 - 60, HEIGHT // 2, (255, 0, 0))
    draw_text("Press R to Restart or ESC to Quit", WIDTH // 2 - 200, HEIGHT // 2 + 100, WHITE)
    pygame.display.flip()
