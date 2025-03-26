import pygame
from player import Player
from config import HEIGHT, WIDTH
pygame.init()


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RPG")

bg = pygame.image.load("background.jpg")
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))

clock = pygame.time.Clock()
all_sprites = pygame.sprite.Group()



player = Player()
all_sprites.add(player)

running = True

while running:
    clock.tick(60)
    screen.blit(bg, (0, 0))

    keys = pygame.key.get_pressed()
    player.update(keys)

    all_sprites.draw(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
