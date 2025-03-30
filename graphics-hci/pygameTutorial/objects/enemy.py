from pygameTutorial.config import *
from pygameTutorial.load_animations import load_animations


class Enemy(pygame.sprite.Sprite):
    def __init__(self, sprite_sheets= None, x = 0, y=HEIGHT // 2, scale_factor=2, speed=2, attack_distance=50, damage=20, hp=70,
                 animations=None):
        super().__init__()
        if sprite_sheets is None:
            self.animations = animations
        else:
            self.animations = load_animations(scale_factor, sprite_sheets)
        self.state = "idle_right"
        self.frame_index = 0
        self.image = self.animations[self.state][self.frame_index]
        self.rect = self.image.get_rect(topleft=(x, y))
        self.mask = pygame.mask.from_surface(self.image)

        self.hp = hp
        self.speed = speed
        self.attack_distance = attack_distance
        self.is_attacking = False
        self.attack_timer = 0
        self.attack_cooldown = 2000
        self.last_direction = "right"
        self.attack_speed = 200
        self.animation_speed = 120
        self.last_update = 0
        self.down = False

        self.is_hurt = False
        self.hurt_timer = 0
        self.damage = damage
        self.hurt_duration = 1000
        self.dead_timer = 500

    def take_damage(self, damage):
        if not self.is_hurt:
            self.hp -= damage
            self.is_hurt = True
            self.hurt_timer = pygame.time.get_ticks()
            self.state = f"hurt_{self.last_direction}"
            self.frame_index = 0

        if self.hp <= 0:
            self.state = f"dead_{self.last_direction}"
            self.frame_index = 0
            self.dead_timer = pygame.time.get_ticks()

    def update(self, player):

        if not player.down:
            now = pygame.time.get_ticks()
            if self.state in ['dead_right', 'dead_left']:
                self.down = True
                if self.frame_index != len(self.animations[self.state]) - 1:
                    if now - self.last_update > self.animation_speed:
                        self.last_update = now
                        self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
                        self.image = self.animations[self.state][self.frame_index]
                else:
                    if now - self.dead_timer > 2000:
                        self.kill()
            else:

                if now - self.hurt_timer > self.hurt_duration:
                    self.is_hurt = False

                if player.rect.centerx < self.rect.centerx and not self.is_attacking and not self.is_hurt:
                    self.rect.x -= self.speed
                    self.state = "move_left"
                    self.last_direction = "left"
                elif player.rect.centerx > self.rect.centerx and not self.is_attacking and not self.is_hurt:
                    self.rect.x += self.speed
                    self.state = "move_right"
                    self.last_direction = "right"

                offset_x = player.rect.x - self.rect.x
                offset_y = player.rect.y - self.rect.y

                if (((self.mask.overlap(player.mask, (offset_x - 50, offset_y))
                     or self.mask.overlap(player.mask, (offset_x + 50, offset_y)))
                    and now - self.attack_timer > self.attack_cooldown) or self.is_attacking) and not self.is_hurt:
                    if self.is_attacking:
                        if self.frame_index >= len(self.animations[self.state]) - 1:
                            if (self.mask.overlap(player.mask, (offset_x - 50, offset_y))
                                                  and self.last_direction == "right") or (self.mask.overlap(player.mask, (offset_x + 50, offset_y))
                                                         and self.last_direction == "left"):
                                player.take_damage(self.damage)
                                self.is_attacking = False
                                self.frame_index = 0
                    else:
                        self.attack_timer = now
                        self.state = "attack_left" if player.rect.centerx < self.rect.centerx else "attack_right"
                        self.is_attacking = True
                        self.frame_index = 0


                if self.is_attacking and not self.is_hurt:
                    if now - self.attack_timer > self.attack_speed:
                        self.attack_timer = now
                        self.frame_index += 1
                        if self.frame_index >= len(self.animations[self.state]):
                            self.is_attacking = False
                            self.state = f"move_{self.last_direction}"
                            self.frame_index = 0
                        self.image = self.animations[self.state][self.frame_index]
                        self.mask = pygame.mask.from_surface(self.image)


                elif now - self.last_update > self.animation_speed:
                    self.last_update = now
                    self.frame_index = (self.frame_index + 1) % len(self.animations[self.state])
                    self.image = self.animations[self.state][self.frame_index]
                    self.mask = pygame.mask.from_surface(self.image)
