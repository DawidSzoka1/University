import pygame

pygame.init()
pygame.mixer.init()
HEIGHT = 1024
WIDTH = 1800
clock = pygame.time.Clock()
font = pygame.font.Font(None, 40)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
SCORE = 0

idle_sheet = pygame.image.load("playerImages/Idle.png").convert_alpha()
walk_sheet = pygame.image.load("playerImages/Walk.png").convert_alpha()
attack_sheet = pygame.image.load("playerImages/Special_Blow_1.png").convert_alpha()
basic_sheet = pygame.image.load("playerImages/Special_Blow_2.png").convert_alpha()
hurt_sheet = pygame.image.load("playerImages/Hurt_1.png").convert_alpha()
dead_sheet = pygame.image.load("playerImages/K.O..png").convert_alpha()

# Definicja podziału sprite sheetów (ilość wierszy, ilość kolumn)
player_sprite_sheets = {
    "idle": (idle_sheet, 1, 7),  # 1 wiersz, 4 kolumny
    "move": (walk_sheet, 1, 10),  # 1 wiersz, 6 kolumn
    "attack": (attack_sheet, 1, 10),# 1 wiersz, 5 kolumn
    "basic": (basic_sheet, 1, 10),
    "hurt": (hurt_sheet, 1, 5),
    "dead": (dead_sheet, 1, 9),
}


enemy_idle = pygame.image.load("enemyImages/stageOne/Idle.png").convert_alpha()
enemy_walk = pygame.image.load("enemyImages/stageOne/Walk.png").convert_alpha()
enemy_hurt = pygame.image.load("enemyImages/stageOne/Hurt.png").convert_alpha()
enemy_attack = pygame.image.load("enemyImages/stageOne/Attack_1.png").convert_alpha()
enemy_dead = pygame.image.load("enemyImages/stageOne/Dead.png").convert_alpha()
enemy_sprite_sheets = {
    "idle": (enemy_idle, 1, 5),
    "move": (enemy_walk, 1, 9),
    "hurt": (enemy_hurt, 1, 2),
    "attack": (enemy_attack, 1, 4),
    "dead": (enemy_dead, 1, 6)
}




STAGE = 1
EnemyPerStage = STAGE * 10
EnemySpawnInterval = 2000
EXPPerStage = STAGE * 5 + 25
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)