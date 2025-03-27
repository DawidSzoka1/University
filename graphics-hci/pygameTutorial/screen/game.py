
from pygameTutorial.config import *
from pygameTutorial.util import draw_text

def game_screen(bg, player, all_sprites, enemy_spawner, score):
    screen.blit(bg, (0, 0))
    keys = pygame.key.get_pressed()
    enemies = enemy_spawner.enemies
    all_sprites.update(keys, enemies)
    all_sprites.draw(screen)
    draw_text(enemy_spawner.message, WIDTH // 2 - 200, 40, BLACK)
    pygame.draw.rect(screen, GRAY, pygame.Rect(10, 30, 400, 50))
    draw_text(f"Aktualna ilosc punktow: {score}", 15, 40, (128,0, 0))
    draw_text(player.message, WIDTH // 2 - 200, 100, BLACK)
    enemy_spawner.update(player)
    enemy_spawner.draw(screen)