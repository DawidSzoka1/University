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

    def update(self, player):
        now = pygame.time.get_ticks()
        if len(self.enemies) < self.max_enemies and now - self.last_spawn_time > self.spawn_interval:
            self.spawn_enemy(player)
            self.last_spawn_time = now

        for enemy in self.enemies:
            enemy.update(player)  # Aktualizujemy wroga, przekazując gracza

    def spawn_enemy(self, player):
        side = random.choice(["left", "right"])
        x_position = 0 if side == "left" else WIDTH
        y_position = random.randint(100, HEIGHT - 100)  # Losowa wysokość

        new_enemy = self.enemy_class(self.sprite_sheets, x_position, y_position)
        new_enemy.player = player  # Przypisujemy gracza jako cel
        self.enemies.add(new_enemy)

    def draw(self, screen):
        self.enemies.draw(screen)