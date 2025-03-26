import pygame
from config import HEIGHT, WIDTH, screen
from pygameTutorial.screen.game import game_screen
from pygameTutorial.screen.menu import menu_screen

pygame.init()

pygame.display.set_caption("RPG")
bg = pygame.image.load("stageBackground/bamboo_bridge.png")
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))

clock = pygame.time.Clock()
all_sprites = pygame.sprite.Group()

running = True

MENU = "menu"
GAME = "game"
OPTIONS = "options"
current_state = MENU


while running:
    clock.tick(60)
    screen.blit(bg, (0, 0))

    if current_state == MENU:
        start_btn, exit_btn = menu_screen()

    elif current_state == GAME:
        game_screen()

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if current_state == MENU:
                if start_btn.collidepoint(mouse_x, mouse_y):
                    current_state = GAME  # Przejście do gry
                if exit_btn.collidepoint(mouse_x, mouse_y):
                    running = False  # Zamknięcie gry

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE and current_state == GAME:
                current_state = MENU  # Powrót do menu

pygame.quit()
