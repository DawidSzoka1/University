import pygame
from config import *


class EnemyStageOne(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.animations = {

        }
        self.state = ""
        self.frame_index = 0
        self.image = self.animations[self.state][self.frame_index]
        self.speed = 2
        self.direction = "right"
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - 80))
        self.last_update = pygame.time.get_ticks()
        self.animation_speed = 120
        self.attack_speed = 60

        self.hp = 50


