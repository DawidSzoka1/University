from pygameTutorial.config import *
from pygameTutorial.util import draw_text

health_bar_width = 300
health_bar_height = 40
def game_screen(bg, player, all_sprites, enemy_spawner, score):
    screen.blit(bg, (0, 0))
    keys = pygame.key.get_pressed()
    enemies = enemy_spawner.enemies
    all_sprites.update(keys, enemies)
    all_sprites.draw(screen)
    draw_text(enemy_spawner.message, WIDTH // 2 - 200, 40, BLACK)
    pygame.draw.rect(screen, GRAY, pygame.Rect(10, 30, 400, 50))
    draw_text(f"Aktualna ilosc punktow: {score}", 15, 40, (128, 0, 0))
    draw_text(player.message, WIDTH // 2 - 200, 100, BLACK)
    enemy_spawner.update(player)
    enemy_spawner.draw(screen)
    current_health_width = (player.hp / player.max_hp) * health_bar_width

    # Ustawienie paska w PRAWYM GÓRNYM ROGU
    x = WIDTH - health_bar_width - 50  # 50 pikseli od prawej krawędzi
    y = 30  # 30 pikseli od góry

    # Rysowanie ramki paska zdrowia
    pygame.draw.rect(screen, (255, 255, 255), (x, y, health_bar_width, health_bar_height), 3)  # Obrys

    # Rysowanie aktualnego zdrowia
    pygame.draw.rect(screen, (255, 0, 0), (x, y, current_health_width, health_bar_height))
