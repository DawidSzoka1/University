from pygameTutorial.config import *
from pygameTutorial.load_animations import load_animations

class Enemy(pygame.sprite.Sprite):
    def __init__(self, sprite_sheets, x, y, scale_factor=2, speed=2, attack_distance=50):
        super().__init__()
        self.animations = load_animations(scale_factor, sprite_sheets)
        self.state = "idle_right"
        self.frame_index = 0
        self.image = self.animations[self.state][self.frame_index]
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

        self.hp = 70
        self.speed = speed
        self.attack_distance = attack_distance
        self.is_attacking = False
        self.attack_timer = 0
        self.attack_cooldown = 2000  # 1 sekunda przerwy między atakami
        self.attack_damage = 20
        self.animation_speed = 120
        self.last_update = 0
        # Zmienne do animacji otrzymywania obrażeń
        self.is_hurt = False
        self.hurt_timer = 0
        self.hurt_duration = 500

    def move_towards(self, player):
        if not self.is_attacking and not self.is_hurt:
            dx = player.rect.centerx - self.rect.centerx
            dy = player.rect.centery - self.rect.centery
            distance = (dx ** 2 + dy ** 2) ** 0.5  # Obliczamy dystans

            if distance > self.attack_distance:  # Jeśli gracz jest dalej niż dystans ataku
                self.rect.x += self.speed if dx > 0 else -self.speed
                self.rect.y += self.speed if dy > 0 else -self.speed
                self.state = "move_left"
            else:
                self.start_attack(player)

    def start_attack(self, player):
        now = pygame.time.get_ticks()
        if now - self.attack_timer > self.attack_cooldown:
            self.attack_timer = now
            self.is_attacking = True
            self.state = "attack"
            self.frame_index = 0

            player.take_damage(self.attack_damage)

    def take_damage(self, damage):
        if not self.is_hurt:
            self.hp -= damage
            self.is_hurt = True
            self.hurt_timer = pygame.time.get_ticks()
            self.state = "hurt_right"
            self.frame_index = 0

            print(f"Przeciwnik traci {damage} HP! Pozostało: {self.hp}")

        if self.hp <= 0:
            print("Przeciwnik pokonany!")
            self.kill()

    def update(self, player):
        """Przeciwnik podąża za graczem i zmienia animację w zależności od kierunku"""

        # Sprawdzenie, czy gracz jest po lewej czy prawej stronie
        if player.rect.centerx < self.rect.centerx:
            self.rect.x -= self.speed  # Ruch w lewo
            self.state = "move_left"  # Ustaw animację ruchu w lewo
        else:
            self.rect.x += self.speed  # Ruch w prawo
            self.state = "move_right"  # Ustaw animację ruchu w prawo

        # Animacja ataku, jeśli przeciwnik blisko gracza
        if self.rect.colliderect(player.rect):
            self.state = "attack_left" if player.rect.centerx < self.rect.centerx else "attack_right"

        # Aktualizacja klatki animacji
        now = pygame.time.get_ticks()
        if now - self.last_update > self.animation_speed:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
            self.image = self.animations[self.state][self.frame_index]
