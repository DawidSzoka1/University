# SWEET EMPIRE 8.0 – Panel dokładnie nad taśmą, w samym środku ekranu
# Nic nie zasłania fabryki ani sklepu!

import pygame
import threading
import random
import time
import math

pygame.init()
WIDTH, HEIGHT = 1400, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SWEET EMPIRE 8.0 – Panel nad taśmą!")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 30)
big_font = pygame.font.Font(None, 70)

# === STANY GRY ===
money = 0
workers_count = 3
worker_speed = 2.0
conveyor_speed = 1.0
customer_freq = 3.0
MAX_QUEUE = 15

conveyor = [None] * 12
conveyor_lock = threading.Lock()
shop_stock = 0
SHOP_MAX = 8

customers = []
lost_customers = 0
global_time = 0

# === WĄTKI ===
def producer():
    while True:
        time.sleep(3.0 / (worker_speed * workers_count))
        with conveyor_lock:
            if conveyor[0] is None:
                conveyor[0] = "candy"

threading.Thread(target=producer, daemon=True).start()

def move_conveyor():
    while True:
        time.sleep(0.8 / conveyor_speed)
        with conveyor_lock:
            global shop_stock
            if conveyor[11] is None or (conveyor[11] == "candy" and shop_stock < SHOP_MAX):
                if conveyor[11] == "candy":
                    shop_stock += 1
                    conveyor[11] = None
                for i in range(11, 0, -1):
                    conveyor[i] = conveyor[i-1]
                conveyor[0] = None

threading.Thread(target=move_conveyor, daemon=True).start()

def spawn_customer():
    while True:
        time.sleep(random.uniform(0.8, 1.8) * customer_freq)
        waiting = [c for c in customers if c['state'] == 'waiting']
        if len(waiting) >= MAX_QUEUE:
            lost_customers += 1
            continue
        customers.append({
            'x': WIDTH + 50,
            'y': 520,
            'patience': 120,
            'state': 'walking_in',
            'anim_time': 0
        })

threading.Thread(target=spawn_customer, daemon=True).start()

# === RYSOWANIE ===
def draw_background():
    screen.fill((135, 206, 235))
    pygame.draw.rect(screen, (100, 80, 60), (0, 500, WIDTH, 300))

def draw_factory():
    blocked = conveyor[0] is not None or shop_stock >= SHOP_MAX
    color = (255, 80, 80) if blocked else (200, 50, 50)
    pygame.draw.rect(screen, color, (80, 280, 380, 220), border_radius=30)
    pygame.draw.rect(screen, (255, 100, 100), (80, 280, 380, 220), 12, border_radius=30)

    for i in range(workers_count):
        x = 140 + i * 110
        pygame.draw.circle(screen, (255, 220, 180), (x, 380), 25)
        pygame.draw.rect(screen, (0, 100, 255), (x-20, 405, 40, 80), border_radius=10)
        anim = math.sin(global_time * 0.01 + i) * 15 if not blocked else 0
        pygame.draw.circle(screen, (255, 150, 50), (int(x + anim), 430), 20)

    if blocked:
        warn = big_font.render("FABRYKA STOI!", True, (255, 50, 50))
        screen.blit(warn, (100, 150))

def draw_conveyor():
    shop_full = shop_stock >= SHOP_MAX
    pygame.draw.rect(screen, (70, 70, 90), (480, 390, 720, 100), border_radius=20)

    angle_speed = 0 if shop_full else global_time * 0.05 * conveyor_speed
    for i in range(13):
        x = 480 + i*60
        for j in range(8):
            dx = math.cos(angle_speed + j*0.8) * 18
            dy = math.sin(angle_speed + j*0.8) * 18
            pygame.draw.circle(screen, (120,120,140), (int(x+dx), 440 + int(dy)), 10)

    with conveyor_lock:
        for i in range(12):
            if conveyor[i] == "candy":
                x = 520 + i * 60
                pygame.draw.circle(screen, (255, 100, 200), (x, 440), 28)
                pygame.draw.circle(screen, (255, 200, 220), (x, 440), 20)

def draw_shop():
    color = (255, 100, 100) if shop_stock >= SHOP_MAX else (100, 200, 100)
    pygame.draw.rect(screen, color, (1180, 280, 180, 220), border_radius=30)
    pygame.draw.rect(screen, (150, 255, 150), (1180, 280, 180, 220), 12, border_radius=30)
    text = font.render("SKLEP", True, (255,255,255))
    screen.blit(text, (1200, 300))

    for i in range(shop_stock):
        row = i // 4
        col = i % 4
        sx = 1195 + col * 35
        sy = 360 + row * 45
        pygame.draw.circle(screen, (255, 150, 100), (sx, sy), 18)
        pygame.draw.circle(screen, (255, 200, 150), (sx, sy), 13)

def update_customers():
    global money, lost_customers, shop_stock
    waiting = [c for c in customers if c['state'] == 'waiting']

    for cust in customers[:]:
        cust['anim_time'] += 1

        if cust['state'] == 'walking_in':
            cust['x'] -= 3
            if cust['x'] <= 1150 - len(waiting) * 45:
                cust['state'] = 'waiting'
                cust['x'] = 1150 - len(waiting) * 45

        elif cust['state'] == 'waiting':
            cust['patience'] -= 0.7
            cust['y'] = 520 + math.sin(cust['anim_time'] * 0.15) * 6
            if cust['patience'] <= 0:
                lost_customers += 1
                customers.remove(cust)
                continue

            if waiting.index(cust) == 0 and shop_stock > 0 and random.random() < 0.07:
                shop_stock -= 1
                money += 25
                cust['state'] = 'leaving_happy'

        elif cust['state'] == 'leaving_happy':
            cust['x'] -= 5
            if cust['x'] < -50:
                customers.remove(cust)

        col = (100, 200, 255) if cust['patience'] > 30 else (255, 100, 100)
        if cust['state'] == 'leaving_happy': col = (100, 255, 100)

        pygame.draw.circle(screen, col, (int(cust['x']), int(cust['y'])), 28)
        pygame.draw.circle(screen, (0,0,0), (int(cust['x']-10), int(cust['y']-8)), 6)
        pygame.draw.circle(screen, (0,0,0), (int(cust['x']+10), int(cust['y']-8)), 6)

        if cust['state'] == 'leaving_happy':
            pygame.draw.arc(screen, (255,255,0), (int(cust['x']-18), int(cust['y']-5), 36, 25), 0.2, 2.9, 8)
        elif cust['patience'] < 30:
            pygame.draw.line(screen, (255,0,0), (int(cust['x']-15), int(cust['y']+15)), (int(cust['x']+15), int(cust['y']+15)), 6)

# === PANEL NAD TAŚMĄ – W SAMYM ŚRODKU ===
def draw_center_panel():
    panel_x = WIDTH // 2 - 200
    panel_y = 20
    pygame.draw.rect(screen, (0, 0, 0, 200), (panel_x, panel_y, 400, 340), border_radius=25)
    pygame.draw.rect(screen, (255, 215, 0), (panel_x, panel_y, 400, 340), 5, border_radius=25)

    title = font.render("SWEET EMPIRE", True, (255,215,0))
    screen.blit(title, (panel_x + 70, panel_y + 15))

    money_text = font.render(f"KASA: ${money:,}", True, (255,255,100))
    screen.blit(money_text, (panel_x + 70, panel_y + 70))

    lines = [
        f"Pracownicy: {workers_count}/4",
        f"Robotnicy: {worker_speed:.1f}x",
        f"Taśma: {conveyor_speed:.1f}x",
        f"Klienci co: {customer_freq:.1f}s",
        f"Sklep: {shop_stock}/{SHOP_MAX}",
        f"Straceni: {lost_customers}"
    ]
    y = panel_y + 130
    for line in lines:
        surf = small_font.render(line, True, (255,255,255))
        screen.blit(surf, (panel_x + 20, y))
        y += 40

    # przyciski + / –
    btn_y = panel_y + 130
    for i in range(4):
        # +
        pygame.draw.rect(screen, (0,200,0), (panel_x + 340, btn_y, 40, 40))
        pygame.draw.rect(screen, (255,255,255), (panel_x + 340, btn_y, 40, 40), 3)
        screen.blit(small_font.render("+", True, (255,255,255)), (panel_x + 355, btn_y + 5))
        # –
        pygame.draw.rect(screen, (200,0,0), (panel_x + 290, btn_y, 40, 40))
        pygame.draw.rect(screen, (255,255,255), (panel_x + 290, btn_y, 40, 40), 3)
        screen.blit(small_font.render("–", True, (255,255,255)), (panel_x + 305, btn_y + 5))
        btn_y += 40

# === GŁÓWNA PĘTLA ===
running = True
while running:
    global_time += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            panel_x = WIDTH // 2 - 200
            panel_y = 20
            btn_y = panel_y + 130
            for i in range(4):
                # +
                if panel_x + 340 <= mx <= panel_x + 380 and btn_y <= my <= btn_y + 40:
                    if i == 0: workers_count = min(4, workers_count + 1)
                    elif i == 1: worker_speed = min(5.0, worker_speed + 0.5)
                    elif i == 2: conveyor_speed = min(5.0, conveyor_speed + 0.5)
                    elif i == 3: customer_freq = max(0.5, customer_freq - 0.5)
                # –
                if panel_x + 290 <= mx <= panel_x + 330 and btn_y <= my <= btn_y + 40:
                    if i == 0: workers_count = max(1, workers_count - 1)
                    elif i == 1: worker_speed = max(0.5, worker_speed - 0.5)
                    elif i == 2: conveyor_speed = max(0.5, conveyor_speed - 0.5)
                    elif i == 3: customer_freq = min(8.0, customer_freq + 0.5)
                btn_y += 40

    draw_background()
    draw_factory()
    draw_conveyor()
    draw_shop()
    update_customers()
    draw_center_panel()   # NAD TAŚMĄ, W ŚRODKU – NIC NIE ZASŁANIA!

    pygame.display.flip()
    clock.tick(60)

pygame.quit()