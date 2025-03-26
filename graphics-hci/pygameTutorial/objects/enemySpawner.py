import random
from pygameTutorial.config import *

class EnemySpawner:
    def __init__(self, enemy_class, sprite_sheets, spawn_interval=2000, max_enemies=10):
        self.enemy_class = enemy_class
        self.sprite_sheets = sprite_sheets
        self.spawn_interval = spawn_interval
        self.max_enemies = max_enemies
        self.enemies = pygame.sprite.Group()
        self.last_spawn_time = pygame.time.get_ticks()

    def update(self):
        now = pygame.time.get_ticks()
        if len(self.enemies) < self.max_enemies and now - self.last_spawn_time > self.spawn_interval:
            self.spawn_enemy()
            self.last_spawn_time = now
        self.enemies.update()

    def spawn_enemy(self):
        side = random.choice(["left", "right"])
        x_position = 0 if side == "left" else WIDTH
        new_enemy = self.enemy_class(self.sprite_sheets, x_position)
        self.enemies.add(new_enemy)

    def draw(self, screen):
        self.enemies.draw(screen)