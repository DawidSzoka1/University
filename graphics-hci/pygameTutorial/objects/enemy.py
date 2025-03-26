import pygame
from pygameTutorial.config import *


class EnemyStageOne(pygame.sprite.Sprite):
    def __init__(self, x, direction):
        super().__init__()
        self.animations = {
            "idle_left": enemyStageOneIdleLeft,
            "idle_right": enemyStageOneIdleRight
        }
        self.direction = direction
        self.state = "idle_right" if direction == "right" else "idle_left"
        self.frame_index = 0
        self.image = self.animations[self.state][self.frame_index]
        self.rect = self.image.get_rect(topleft=(x, HEIGHT - 100))
        self.speed = 1 if direction == "right" else -1
        self.last_update = pygame.time.get_ticks()
        self.animation_speed = 60

    def update(self, keys):
        self.rect.x += self.speed
        now = pygame.time.get_ticks()
        if now - self.last_update > self.animation_speed:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
            self.image = self.animations[self.state][self.frame_index]