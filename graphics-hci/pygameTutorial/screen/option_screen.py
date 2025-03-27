from pygameTutorial.config import *
from pygameTutorial.objects.player import Player
from pygameTutorial.util import draw_text

bg = pygame.image.load("stageBackground/bamboo_bridge.png")
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
player = Player(player_sprite_sheets, border_x=(100, WIDTH - 100), start=(WIDTH //2 , HEIGHT // 2  + 200))
all_sprites = pygame.sprite.Group()
all_sprites.add(player)
def option_screen():
    screen.blit(bg, (0, 0))

    menu_button = pygame.Rect(WIDTH // 2 - 300, HEIGHT // 2 - 180, 250, 50)
    pygame.draw.rect(screen, WHITE, menu_button)
    draw_text("Powrot do menu", WIDTH // 2 - 280, HEIGHT // 2 - 165, BLACK)

    draw_text("a - ruch w lewo", WIDTH // 2 - 300, HEIGHT // 2 - 105, BLACK)
    draw_text("d - ruch w prawo", WIDTH // 2 - 300, HEIGHT // 2 - 55, BLACK)
    draw_text("q - atak  podstawowy w kierunku ruchu", WIDTH // 2 - 300, HEIGHT // 2 - 5, BLACK)
    draw_text("e - atak specjalny w kierunku ruchu", WIDTH // 2 - 300, HEIGHT // 2 + 45, BLACK)
    draw_text("esc - powrot do menu", WIDTH // 2 - 300, HEIGHT // 2 + 95, BLACK)
    keys = pygame.key.get_pressed()
    all_sprites.update(keys, [])
    all_sprites.draw(screen)
    return menu_button