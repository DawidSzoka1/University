import random

from pygameTutorial.config import *
from pygameTutorial.util import enemy_sheets

class EnemySpawner:
    def __init__(self, enemy_class, sprite_sheets,
                 spawn_interval=2000, max_enemies=10,
                 enemy_hp=70, enemy_damage=20):
        self.enemy_hp = enemy_hp
        self.enemy_damage = enemy_damage
        self.enemy_class = enemy_class
        self.sprite_sheets = sprite_sheets
        self.spawn_interval = spawn_interval
        self.message = ""
        self.max_enemies = max_enemies
        self.count = 0
        self.enemies = pygame.sprite.Group()
        self.last_spawn_time = pygame.time.get_ticks()

    def update(self, player):
        now = pygame.time.get_ticks()
        if self.count < self.max_enemies and now - self.last_spawn_time > self.spawn_interval:
            self.spawn_enemy(player)
            self.count += 1
            self.last_spawn_time = now

        for enemy in self.enemies:
            enemy.update(player)

    def spawn_enemy(self, player):
        side = random.choice(["left", "right"])
        x_position = 0 if side == "left" else WIDTH
        y_position = HEIGHT // 2
        skin = random.choice(["skinBasic", "skinArcher", "skinCommander"])
        if skin == "skinArcher":
            amount = [9, 8, 3, 5, 5]
        elif skin == "skinBasic":
            amount = [6, 9, 3, 4, 6]
        else:
            amount = [5, 9, 2, 4, 6]
        new_enemy = self.enemy_class(enemy_sheets(skin, amount), x_position, y_position,
                                     damage=self.enemy_damage, hp=self.enemy_hp)
        new_enemy.player = player
        self.enemies.add(new_enemy)

    def draw(self, screen):
        self.enemies.draw(screen)
