from pygameTutorial.config import *
from pygameTutorial.util import draw_text, display_skin


def pick_skin(picked):
    draw_text("Wybierz skorke", WIDTH / 2, HEIGHT / 2 - 200, BLACK)
    check_img = pygame.image.load("playerImages/check-mark.png")
    check_img = pygame.transform.scale(check_img, (32, 32))

    green_skin = pygame.Rect(WIDTH // 2 - 400, HEIGHT // 2 + 20, 240, 50)
    pygame.draw.rect(screen, GRAY, green_skin)
    draw_text("Zielony skin", WIDTH // 2 - 390, HEIGHT / 2 + 35, WHITE)
    display_skin("skinGreen", WIDTH // 2 - 350, HEIGHT // 2 + 100)
    if picked == "green":
        screen.blit(check_img, (WIDTH // 2 - 210 , HEIGHT // 2 + 27))


    red_skin = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 20, 240, 50)
    pygame.draw.rect(screen, GRAY, red_skin)
    draw_text("Czerwony skin", WIDTH // 2 - 95, HEIGHT / 2 + 35, WHITE)
    display_skin("skinRed", WIDTH // 2 - 50, HEIGHT // 2 + 100)
    if picked == "red":
        screen.blit(check_img, (WIDTH // 2 + 110, HEIGHT // 2 + 27))


    purple_skin = pygame.Rect(WIDTH // 2 + 200, HEIGHT // 2 + 20, 240, 50)
    pygame.draw.rect(screen, GRAY, purple_skin)
    draw_text("Fioletowy skin", WIDTH // 2 + 205, HEIGHT / 2 + 35, WHITE)
    display_skin("skinPurple", WIDTH // 2 + 250, HEIGHT // 2 + 100)
    if picked == "purple":
        screen.blit(check_img, (WIDTH // 2 + 400 , HEIGHT // 2 + 27))

    return green_skin, red_skin, purple_skin
