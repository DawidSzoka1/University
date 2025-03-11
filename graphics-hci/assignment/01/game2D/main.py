import pygame
import random
from bullet import *

# Konfiguracja
WIDTH, HEIGHT = 800, 600
FPS = 60

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Inicjalizacja gry
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Strzelanka kosmiczna")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Wczytywanie obrazków
player_img = pygame.image.load("images/player.png")
enemy_img = pygame.image.load("images/enemy.png")
bad_target_img = pygame.image.load("images/bad_target.png")
powerup_img = pygame.image.load("images/powerup.jpg")
boss_img = pygame.image.load("images/boss.png")
boss_bullet_img = pygame.image.load("images/boss_bullet.jpg")


# Klasy
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.transform.scale(player_img, (50, 50))
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - 60))
        self.powered_up = False

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= 5
        if keys[pygame.K_RIGHT] and self.rect.right < WIDTH:
            self.rect.x += 5

    def shoot(self):
        if self.powered_up:
            bullets.add(Bullet(self.rect.centerx - 10, self.rect.top))
            bullets.add(Bullet(self.rect.centerx + 10, self.rect.top))
        else:
            bullets.add(Bullet(self.rect.centerx, self.rect.top))


class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.transform.scale(enemy_img, (40, 40))
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.rect.y += 2
        if self.rect.top > HEIGHT:
            self.rect.y = random.randint(-100, -40)
            self.rect.x = random.randint(40, WIDTH - 40)


class BadTarget(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.transform.scale(bad_target_img, (40, 40))
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.rect.y += 2
        if self.rect.top > HEIGHT:
            self.rect.y = random.randint(-100, -40)
            self.rect.x = random.randint(40, WIDTH - 40)


class PowerUp(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.transform.scale(powerup_img, (30, 30))
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.rect.y += 3
        if self.rect.top > HEIGHT:
            self.kill()


def game_over_screen():
    screen.fill(BLACK)
    game_over_text = font.render("Game Over", True, WHITE)
    score_text = font.render(f"Twój wynik: {score}", True, WHITE)
    restart_text = font.render("Naciśnij ENTER, aby zagrać ponownie", True, WHITE)

    screen.blit(game_over_text, (WIDTH // 2 - 50, HEIGHT // 2 - 50))
    screen.blit(score_text, (WIDTH // 2 - 70, HEIGHT // 2))
    screen.blit(restart_text, (WIDTH // 2 - 150, HEIGHT // 2 + 50))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False


# Grupy
player = Player()
players = pygame.sprite.Group(player)
bullets = pygame.sprite.Group()
enemies = pygame.sprite.Group()
bad_targets = pygame.sprite.Group()
powerups = pygame.sprite.Group()

# Tworzenie wrogów i złych celów
for _ in range(5):
    enemies.add(Enemy(random.randint(40, WIDTH - 40), random.randint(-100, -40)))
for _ in range(2):
    bad_targets.add(BadTarget(random.randint(40, WIDTH - 40), random.randint(-100, -40)))

score = 0
running = True
powerup_timer = 0
while running:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player.shoot()

    # Losowe pojawianie się ulepszeń
    if random.randint(1, 500) == 1:
        powerups.add(PowerUp(random.randint(40, WIDTH - 40), -40))

    # Aktualizacja
    players.update()
    bullets.update()
    enemies.update()
    bad_targets.update()
    powerups.update()

    # Kolizje
    for bullet in bullets:
        hits = pygame.sprite.spritecollide(bullet, enemies, True)
        if hits:
            bullet.kill()
            score += 10
            enemies.add(Enemy(random.randint(40, WIDTH - 40), random.randint(-100, -40)))
        bad_hits = pygame.sprite.spritecollide(bullet, bad_targets, True)
        if bad_hits:
            game_over_screen()
            score = 0
            bullets.empty()
            enemies.empty()
            bad_targets.empty()
            powerups.empty()
            for _ in range(5):
                enemies.add(Enemy(random.randint(40, WIDTH - 40), random.randint(-100, -40)))
            for _ in range(2):
                bad_targets.add(BadTarget(random.randint(40, WIDTH - 40), random.randint(-100, -40)))

    # Sprawdzenie kolizji z ulepszeniem
    if pygame.sprite.spritecollide(player, powerups, True):
        player.powered_up = True
        powerup_timer = pygame.time.get_ticks()

    if player.powered_up and pygame.time.get_ticks() - powerup_timer > 5000:
        player.powered_up = False

    # Rysowanie
    players.draw(screen)
    bullets.draw(screen)
    enemies.draw(screen)
    bad_targets.draw(screen)
    powerups.draw(screen)

    score_text = font.render(f"Punkty: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
