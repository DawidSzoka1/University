import pygame
from pygameTutorial.config import *

class Player(pygame.sprite.Sprite):
    def __init__(self, sprite_sheets, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.animations = self.load_animations(sprite_sheets)
        self.state = "idle_right"
        self.frame_index = 0
        self.image = self.animations[self.state][self.frame_index]
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.speed = 20
        self.last_update = pygame.time.get_ticks()
        self.animation_speed = 100

        self.attack_speed = 50
        self.is_attacking = False
        self.attack_timer = 0
        self.hp = 100
        self.exp = 0
        self.last_direction = "right"  # Śledzenie ostatniego kierunku

    def load_animations(self, sprite_sheets):
        """Wczytuje animacje postaci z wielu sprite sheetów."""
        animations = {}
        for action, (sprite_sheet, rows, cols) in sprite_sheets.items():
            frame_width = sprite_sheet.get_width() // cols
            frame_height = sprite_sheet.get_height() // rows
            frames = []
            for row in range(rows):
                for col in range(cols):
                    frame = sprite_sheet.subsurface(pygame.Rect(col * frame_width, row * frame_height, frame_width, frame_height))
                    scaled_frame = pygame.transform.scale(frame, (frame_width * self.scale_factor, frame_height * self.scale_factor))
                    frames.append(scaled_frame)

            animations[f"{action}_right"] = frames
            animations[f"{action}_left"] = [pygame.transform.flip(img, True, False) for img in frames]

        return animations

    def update(self, keys):
        if not self.is_attacking:  # Ruch tylko jeśli nie atakujemy
            if keys[pygame.K_a] and self.rect.x > 0:
                self.rect.x -= self.speed
                self.state = "move_left"
                self.last_direction = "left"
            elif keys[pygame.K_d] and self.rect.x < WIDTH - self.rect.width:
                self.rect.x += self.speed
                self.state = "move_right"
                self.last_direction = "right"
            else:
                self.state = f"idle_{self.last_direction}"

        # Atak podstawowy
        if keys[pygame.K_q] and not self.is_attacking:
            self.start_attack(f"attack_{self.last_direction}")

        if keys[pygame.K_e] and not self.is_attacking:
            self.start_attack(f"basic_{self.last_direction}")

        now = pygame.time.get_ticks()

        # Animacja ataku
        if self.is_attacking:
            if now - self.attack_timer > self.attack_speed:
                self.attack_timer = now
                self.frame_index += 1
                if self.frame_index >= len(self.animations[self.state]):
                    self.is_attacking = False  # Koniec ataku
                    self.state = f"idle_{self.last_direction}"  # Powrót do idle w kierunku ruchu
                    self.frame_index = 0
                self.image = self.animations[self.state][self.frame_index]

        # Animacja ruchu
        elif now - self.last_update > self.animation_speed:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
            self.image = self.animations[self.state][self.frame_index]

    def start_attack(self, animation):
        """Rozpoczyna animację ataku w ostatnim kierunku ruchu"""
        attack_state = f"{animation}"
        self.is_attacking = True
        self.state = attack_state
        self.frame_index = 0
        self.attack_timer = pygame.time.get_ticks()
