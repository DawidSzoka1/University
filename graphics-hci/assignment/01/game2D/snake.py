import pygame
import random

pygame.init()

# Ustawienia okna gry
win = pygame.display.set_mode((500, 540))
pygame.display.set_caption("Pierwsza gra")

# Pozycja i rozmiar zielonego kwadratu
x, y = 0, 40
szerokosc, wysokosc = 20, 20
krok = 20
kolory = [(0, 255, 0), (255, 255, 0), (0, 0, 255)]  # Zielony -> Żółty -> Niebieski
indeks_koloru = 0

# Czerwone kwadraty
czerwone_kwadraty = []
for _ in range(10):
    czerwone_kwadraty.append((random.randint(0, 480) // 20 * 20, random.randint(0, 520) // 20 * 20))

score = 0
font = pygame.font.SysFont("Arial", 24)
run = True

while run:
    pygame.time.delay(50)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                indeks_koloru = (indeks_koloru + 1) % 3

    # Obsługa sterowania ASDW
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        x -= krok
    if keys[pygame.K_d]:
        x += krok
    if keys[pygame.K_w]:
        y -= krok
    if keys[pygame.K_s]:
        y += krok

    # Sprawdzenie kolizji ze ścianami
    if x < 0 or x + szerokosc > 500 or y < 0 or y + wysokosc > 540:
        print("Game Over")
        run = False
        continue

    # Sprawdzenie kolizji z czerwonymi kwadratami
    nowa_lista = []
    for cx, cy in czerwone_kwadraty:
        if x == cx and y == cy:
            score += 1
        else:
            nowa_lista.append((cx, cy))
    czerwone_kwadraty = nowa_lista

    # Czyszczenie ekranu
    win.fill((0, 0, 0))

    # Rysowanie czerwonych kwadratów
    for cx, cy in czerwone_kwadraty:
        pygame.draw.rect(win, (255, 0, 0), (cx, cy, szerokosc, wysokosc))

    # Rysowanie zielonego kwadratu
    pygame.draw.rect(win, kolory[indeks_koloru], (x, y, szerokosc, wysokosc))

    # Wyświetlenie aktualnej pozycji i wyniku
    text = font.render(f"Pozycja: ({x}, {y})  Score: {score}", True, (255, 255, 255))
    win.blit(text, (10, 10))

    pygame.display.update()

pygame.quit()