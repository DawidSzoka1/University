import pygame
from config import screen, font
from pygameTutorial.load_animations import load_animations


def draw_text(text, x, y, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def player_sheets(SKINCOLOR):
    idle_sheet = pygame.image.load(f"playerImages/{SKINCOLOR}/Idle.png").convert_alpha()
    walk_sheet = pygame.image.load(f"playerImages/{SKINCOLOR}/Walk.png").convert_alpha()
    attack_sheet = pygame.image.load(f"playerImages/{SKINCOLOR}/Special_Blow_1.png").convert_alpha()
    basic_sheet = pygame.image.load(f"playerImages/{SKINCOLOR}/Special_Blow_2.png").convert_alpha()
    hurt_sheet = pygame.image.load(f"playerImages/{SKINCOLOR}/Hurt_1.png").convert_alpha()
    dead_sheet = pygame.image.load(f"playerImages/{SKINCOLOR}/K.O..png").convert_alpha()

    return {
        "idle": (idle_sheet, 1, 7),
        "move": (walk_sheet, 1, 10),
        "attack": (attack_sheet, 1, 10),
        "basic": (basic_sheet, 1, 10),
        "hurt": (hurt_sheet, 1, 5),
        "dead": (dead_sheet, 1, 9)
    }

def enemy_sheets(SKINTYPE, amount):
    enemy_idle = pygame.image.load(f"enemyImages/{SKINTYPE}/Idle.png").convert_alpha()
    enemy_walk = pygame.image.load(f"enemyImages/{SKINTYPE}/Walk.png").convert_alpha()
    enemy_hurt = pygame.image.load(f"enemyImages/{SKINTYPE}/Hurt.png").convert_alpha()
    enemy_attack = pygame.image.load(f"enemyImages/{SKINTYPE}/Attack_1.png").convert_alpha()
    enemy_dead = pygame.image.load(f"enemyImages/{SKINTYPE}/Dead.png").convert_alpha()
    return {
        "idle": (enemy_idle, 1, amount[0]),
        "move": (enemy_walk, 1, amount[1]),
        "hurt": (enemy_hurt, 1, amount[2]),
        "attack": (enemy_attack, 1, amount[3]),
        "dead": (enemy_dead, 1, amount[4])
    }

def display_skin(skin, x, y):
    screen.blit(pygame.image.load(f"playerImages/{skin}/img_1.png"), (x, y))
