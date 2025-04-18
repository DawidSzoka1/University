import pygame.mixer

from pygameTutorial.load_animations import load_animations
from pygameTutorial.config import *


class Player(pygame.sprite.Sprite):
    def __init__(self, sprite_sheets, scale_factor=2,
                 border_x=(0, WIDTH),
                 start=(WIDTH // 2, HEIGHT // 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.animations = load_animations(scale_factor, sprite_sheets)
        self.max_hp = 100
        self.state = "idle_right"
        self.frame_index = 0
        self.image = self.animations[self.state][self.frame_index]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect(center=start)
        self.speed = 20
        self.last_update = pygame.time.get_ticks()
        self.animation_speed = 120
        self.basic_attack_sound = pygame.mixer.Sound("sound/basic_attack.mp3")
        self.special_attack_sound = pygame.mixer.Sound("sound/special_attack.mp3")
        self.basic_attack_sound.set_volume(0.15)
        self.take_damage_sound = pygame.mixer.Sound("sound/take_damage_sound.mp3")
        self.take_damage_sound.set_volume(0.1)
        self.dead_sound = pygame.mixer.Sound("sound/dead_sound.mp3")
        self.dead_sound.set_volume(0.4)
        self.attack_damage = 20
        self.special_attack_sound.set_volume(0.15)
        self.reload_sound = pygame.mixer.Sound("sound/reload.mp3")
        self.is_reloading = False
        self.border_x = border_x
        self.attack_speed = 50
        self.is_attacking = False
        self.attack_timer = 0
        self.hp = 100
        self.eCooldown = 3000
        self.lastE = 0
        self.last_direction = "right"
        self.message = ""
        self.alive = True
        self.is_hurt = False
        self.hurt_timer = 0
        self.down = False
        self.hurt_duration = 500
        self.dead_timer = 0

    def update(self, keys, enemies):
            now = pygame.time.get_ticks()
            if self.state in ["dead_right", "dead_left"]:
                self.down = True
                if self.frame_index != len(self.animations[self.state]) - 1:
                    if now - self.last_update > self.animation_speed:
                        self.last_update = now
                        self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
                        self.image = self.animations[self.state][self.frame_index]
                if self.frame_index == 2:
                    self.dead_sound.play()
                else:
                    if now - self.dead_timer > 2000:
                        self.kill()
                        self.alive = False
            else:
                if now - self.hurt_timer > self.hurt_duration:
                    self.is_hurt = False
                if self.is_hurt and self.frame_index == 3:
                    self.take_damage_sound.play()
                if not self.is_attacking and not self.is_hurt:
                    if keys[pygame.K_a] and self.rect.x > self.border_x[0]:
                        self.rect.x -= self.speed
                        self.state = "move_left"
                        self.last_direction = "left"
                    elif keys[pygame.K_d] and self.rect.x < self.border_x[1] - self.rect.width:
                        self.rect.x += self.speed
                        self.state = "move_right"
                        self.last_direction = "right"

                    else:
                        self.state = f"idle_{self.last_direction}"
                if self.is_attacking:
                    for enemy in enemies:
                        offset_x = enemy.rect.x - self.rect.x
                        offset_y = enemy.rect.y - self.rect.y
                        if ((self.mask.overlap(enemy.mask, (offset_x - 50, offset_y))
                             and self.last_direction == "right") or (self.mask.overlap(enemy.mask, (offset_x + 50, offset_y))
                                                                     and self.last_direction == "left")) and not enemy.down:
                            enemy.take_damage(self.attack_damage)

                if keys[pygame.K_q] and not self.is_attacking and not self.is_hurt:
                    self.start_attack(f"attack_{self.last_direction}")
                    self.basic_attack_sound.play()
                    self.attack_damage = 20

                remaining_cooldown = max(0, (self.eCooldown - (now - self.lastE)) // 1000)
                if self.is_reloading:
                    self.reload_sound.set_volume(0.1)
                    self.reload_sound.play()
                    self.is_reloading = False

                if remaining_cooldown > 0:
                    self.message = f"E jest na cooldown. Poczekaj {remaining_cooldown}s"

                else:
                    self.message = ""  # Usuwanie wiadomości, gdy cooldown się skończy

                if keys[pygame.K_e] and not self.is_attacking and remaining_cooldown == 0 and not self.is_hurt:
                    self.special_attack_sound.play()
                    self.start_attack(f"basic_{self.last_direction}")
                    self.lastE = now
                    self.attack_damage = 50
                    clock.tick(100000)
                    self.is_reloading = True

                if self.is_attacking:
                    if now - self.attack_timer > self.attack_speed:
                        self.attack_timer = now
                        self.frame_index += 1
                        if self.frame_index >= len(self.animations[self.state]):
                            self.is_attacking = False
                            self.state = f"idle_{self.last_direction}"
                            self.frame_index = 0
                        self.image = self.animations[self.state][self.frame_index]
                        self.mask = pygame.mask.from_surface(self.image)


                elif now - self.last_update > self.animation_speed:
                    self.last_update = now
                    self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
                    self.image = self.animations[self.state][self.frame_index]
                    self.mask = pygame.mask.from_surface(self.image)

    def start_attack(self, animation):

        attack_state = f"{animation}"
        self.is_attacking = True
        self.state = attack_state
        self.frame_index = 0
        self.attack_timer = pygame.time.get_ticks()


    def jump(self):
        if self.on_ground:
            self.vel_y = self.jump_power
            self.on_ground = False
            self.state = f"jump_{self.last_direction}"
            self.frame_index = 0

    def take_damage(self, damage):
            if not self.is_hurt:
                self.hp -= damage
                self.is_hurt = True
                self.hurt_timer = pygame.time.get_ticks()
                self.state = f"hurt_{self.last_direction}"
                self.frame_index = 0


            if self.hp <= 0:
                self.dead_timer = pygame.time.get_ticks()
                self.state = f"dead_{self.last_direction}"

