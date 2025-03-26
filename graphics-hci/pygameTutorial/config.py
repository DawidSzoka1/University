from load_img import load_images
import pygame

pygame.init()
HEIGHT = 1024
WIDTH = 1800

font = pygame.font.Font(None, 40)
screen = pygame.display.set_mode((WIDTH, HEIGHT))

player_walk = load_images("playerImages/move/sprite_", 10)
player_idle = load_images("playerImages/idle/sprite_", 7)
player_special_attack = load_images("playerImages/attack_special/sprite_", 10)
player_basic_attack = load_images("playerImages/attack_basic/sprite_", 10)

enemyStageOneIdleRight = load_images("enemyImages/stageOne/idle/sprite_", 4, 12)
enemyStageOneIdleLeft = load_images("enemyImages/stageOne/idle/sprite_", 4, 8)

STAGE = 1
EnemyPerStage = STAGE * 10
EnemySpawnInterval = 2000
EXPPerStage = STAGE * 5 + 25


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)