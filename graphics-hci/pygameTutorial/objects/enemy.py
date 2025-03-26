from pygameTutorial.load_animations import load_animations
from pygameTutorial.config import *


class Enemy(pygame.sprite.Sprite):
    def __init__(self, sprite_sheets, x):
        super().__init__()
        self.animations = load_animations(2, sprite_sheets)
        self.state = "move_right" if x == 0 else "move_left"
        self.frame_index = 0
        self.image = self.animations[self.state][self.frame_index]
        self.rect = self.image.get_rect(topleft=(x, HEIGHT //2))
        self.speed = 5 if self.state == "move_right" else -5
        self.last_update = pygame.time.get_ticks()
        self.animation_speed = 120

    def update(self):
        self.rect.x += self.speed
        now = pygame.time.get_ticks()
        if now - self.last_update > self.animation_speed:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
            self.image = self.animations[self.state][self.frame_index]
