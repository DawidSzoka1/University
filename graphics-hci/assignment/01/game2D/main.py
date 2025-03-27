import pygame

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game 2D")

player_color = (0, 255, 0)
player_size = 50
player_x = WIDTH // 2
player_y = HEIGHT - player_size
player_speed = 5

running = True
while running:
    pygame.time.delay(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
        player_x += player_speed


    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, player_color, (player_x, player_y, player_size, player_size))
    pygame.display.update()

pygame.quit()