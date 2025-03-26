import pygame
from pygameTutorial.config import *
from pygameTutorial.objects.player import Player
from pygameTutorial.objects.enemy import EnemyStageOne


bg = pygame.image.load('stageBackground/sky_bridge.png')
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
clock = pygame.time.Clock()
all_sprites = pygame.sprite.Group()
player = Player(player_sprite_sheets)
all_sprites.add(player)

def game_screen():
    screen.blit(bg, (0, 0))
    keys = pygame.key.get_pressed()
    all_sprites.update(keys)
    all_sprites.draw(screen)
