import pygame
from config import HEIGHT, WIDTH, screen, clock, player_sprite_sheets, enemy_sprite_sheets
from pygameTutorial.screen.game import game_screen
from pygameTutorial.objects.player import Player
from pygameTutorial.objects.enemy import Enemy
from pygameTutorial.objects.enemySpawner import EnemySpawner
from pygameTutorial.screen.menu import menu_screen
from pygameTutorial.screen.option_screen import option_screen
from pygameTutorial.screen.gameOver import game_over_screen

def start_game(enemies):
    score = 0
    game_bg = pygame.image.load('stageBackground/sky_bridge.png')
    game_bg = pygame.transform.scale(game_bg, (WIDTH, HEIGHT))
    enemy_spawner = EnemySpawner(Enemy, enemy_sprite_sheets, max_enemies=enemies)
    all_sprites = pygame.sprite.Group()
    player = Player(player_sprite_sheets)
    all_sprites.add(player)

    return game_bg, player, all_sprites, enemy_spawner, score

def game_loop():
    pygame.init()
    pygame.mixer.init()
    first_play = True

    lose_screen = pygame.image.load("stageBackground/lose_screen.png")
    lose_screen = pygame.transform.scale(lose_screen, (WIDTH, HEIGHT))
    win_screen = pygame.image.load("stageBackground/win_screen.png")
    win_screen = pygame.transform.scale(win_screen, (WIDTH, HEIGHT))
    pygame.mixer.music.load('sound/menu_sound.mp3')
    pygame.mixer.music.set_volume(0.2)
    pygame.mixer.music.play(-1)
    pygame.display.set_caption("RPG")
    bg = pygame.image.load("stageBackground/bamboo_bridge.png")
    bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
    max_enemies = 10
    game_bg, player, all_sprites, enemy_spawner, score = start_game(max_enemies)
    running = True

    MENU = "menu"
    GAME = "game"
    OVER = "over"
    OPTIONS = "options"
    current_state = MENU
    over_message = ""
    over_bg = ""
    while running:
        clock.tick(60)
        screen.blit(bg, (0, 0))
        if current_state != OVER:
            if current_state == MENU:
                start_btn, exit_btn, option_btn = menu_screen()
            elif current_state == GAME:
                if first_play:
                    first_play = False
                    game_bg, player, all_sprites, enemy_spawner, score = start_game(max_enemies)
                    game_screen( game_bg, player, all_sprites, enemy_spawner, score)
                else:
                    game_screen(game_bg, player, all_sprites, enemy_spawner, score)
                score = (enemy_spawner.count - len(enemy_spawner.enemies)) * 20
                if max_enemies == enemy_spawner.count and len(enemy_spawner.enemies) == 0:
                    current_state = OVER
                    over_message = "You won GG next time will be harder"
                    max_enemies  *= 2
                    over_bg = win_screen
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load('sound/win_song.mp3')
                    pygame.mixer.music.set_volume(0.2)
                    pygame.mixer.music.play(-1)

                if not player.alive:
                    current_state = OVER
                    over_message = "Game Over You Lost"
                    over_bg = lose_screen
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load('sound/lose_song.mp3')
                    pygame.mixer.music.set_volume(0.2)
                    pygame.mixer.music.play(-1)
            elif current_state == OPTIONS:
                menu_button = option_screen()
        if  current_state == OVER:
            current_state = OVER
            game_over_screen(score, over_message, over_bg)
            first_play = True

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if current_state == MENU:
                    if start_btn.collidepoint(mouse_x, mouse_y):
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load("sound/first_stade.mp3")
                        pygame.mixer.music.set_volume(0.2)
                        pygame.mixer.music.play(-1)
                        current_state = GAME  # Przejście do gry
                    if option_btn.collidepoint(mouse_x, mouse_y):
                        current_state = OPTIONS
                    if exit_btn.collidepoint(mouse_x, mouse_y):
                        running = False  # Zamknięcie gry


            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE and (current_state == GAME
                                                     or current_state == OPTIONS):
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load('sound/menu_sound.mp3')
                    pygame.mixer.music.set_volume(0.2)
                    pygame.mixer.music.play(-1)
                    current_state = MENU  # Powrót do menu
                if current_state == OVER:
                    if event.key == pygame.K_r:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load('sound/menu_sound.mp3')
                        pygame.mixer.music.set_volume(0.2)
                        pygame.mixer.music.play(-1)
                        current_state = MENU
                    if event.key == pygame.K_ESCAPE:
                        running = False

if __name__ == '__main__':
    game_loop()
    pygame.quit()
