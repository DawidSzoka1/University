import pygame
from pygameTutorial.config import *
from pygameTutorial.objects.player import Player
from pygameTutorial.objects.enemy import Enemy
from pygameTutorial.objects.enemySpawner import EnemySpawner
from pygameTutorial.util import draw_text

bg = pygame.image.load('stageBackground/sky_bridge.png')
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
enemy_spawner = EnemySpawner(Enemy, enemy_sprite_sheets)
all_sprites = pygame.sprite.Group()
player = Player(player_sprite_sheets)
all_sprites.add(player)

def game_screen():
    screen.blit(bg, (0, 0))
    keys = pygame.key.get_pressed()
    all_sprites.update(keys)
    all_sprites.draw(screen)
    draw_text(player.message, WIDTH // 2 - 200, 100, BLACK)
    enemy_spawner.update()
    enemy_spawner.draw(screen)
