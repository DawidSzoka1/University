import pygame
import math
import random
import time
import threading
import multiprocessing
from multiprocessing import Manager, Process

# -----------------------------
# Ustawienia
# -----------------------------
N = 5
THINK = (0.5, 2.0)
EAT = (0.3, 1.2)
FPS = 60

STATE_THINK = 0
STATE_HUNGRY = 1
STATE_EAT = 2

COLORS = {
    STATE_THINK: (170, 170, 170),
    STATE_HUNGRY: (255, 180, 0),
    STATE_EAT: (0, 200, 70),
}

# -----------------------------
# Symulacje
# -----------------------------

def philosopher_thread(i, states, run_event, locks):
    left = locks[i]
    right = locks[(i+1) % N]

    # deadlock prevention
    first = left if i % 2 == 0 else right
    second = right if i % 2 == 0 else left

    while run_event.is_set():
        states[i] = STATE_THINK
        time.sleep(random.uniform(*THINK))

        states[i] = STATE_HUNGRY
        with first:
            with second:
                states[i] = STATE_EAT
                time.sleep(random.uniform(*EAT))


def philosopher_process(i, states, locks, run_flag):
    left = locks[i]
    right = locks[(i+1) % N]

    first = left if i % 2 == 0 else right
    second = right if i % 2 == 0 else left

    while run_flag.value:
        states[i] = STATE_THINK
        time.sleep(random.uniform(*THINK))

        states[i] = STATE_HUNGRY
        with first:
            with second:
                states[i] = STATE_EAT
                time.sleep(random.uniform(*EAT))


# -----------------------------
# Animacja Pygame
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((900, 900))
pygame.display.set_caption("Problem pięciu filozofów — animacja")

font = pygame.font.SysFont("Arial", 26)
clock = pygame.time.Clock()

# pozycje w okręgu
positions = []
for i in range(N):
    angle = 2 * math.pi * i / N
    x = 450 + 300 * math.cos(angle)
    y = 450 + 300 * math.sin(angle)
    positions.append((int(x), int(y)))

# stan początkowy
states = [0] * N

# global thread/process refs
threads = []
processes = []
locks = [threading.Lock() for _ in range(N)]
run_event = threading.Event()
run_flag = None
mp_manager = None

mode = "none"  # "seq" / "thread" / "mp"
running_sim = False


# -----------------------------
# WERSJA SEKWENCYJNA
# -----------------------------
def sequential_runner():
    global states, running_sim
    while running_sim:
        for i in range(N):
            states[i] = STATE_THINK
            time.sleep(random.uniform(*THINK))
            states[i] = STATE_EAT
            time.sleep(random.uniform(*EAT))
            states[i] = STATE_THINK


# -----------------------------
# Uruchamianie trybów
# -----------------------------

def start_sequential():
    global running_sim, threads, mode
    running_sim = True
    mode = "seq"
    t = threading.Thread(target=sequential_runner, daemon=True)
    threads.append(t)
    t.start()


def start_threading():
    global running_sim, run_event, threads, mode, locks, states

    running_sim = True
    mode = "thread"

    # nowe locki tylko dla threadingu
    locks = [threading.Lock() for _ in range(N)]

    run_event.set()
    for i in range(N):
        t = threading.Thread(
            target=philosopher_thread,
            args=(i, states, run_event, locks),
            daemon=True
        )
        threads.append(t)
        t.start()


def start_multiprocessing():
    global running_sim, processes, mp_manager, run_flag, mode, states, locks

    running_sim = True
    mode = "mp"

    mp_manager = Manager()
    states = mp_manager.list([0] * N)

    # GLOBALNE locki dla MP — najważniejsza poprawka
    locks = [mp_manager.Lock() for _ in range(N)]

    run_flag = mp_manager.Value('b', True)

    for i in range(N):
        p = Process(
            target=philosopher_process,
            args=(i, states, locks, run_flag)
        )
        processes.append(p)
        p.start()


def stop_all():
    global running_sim, run_event, run_flag, locks
    running_sim = False

    run_event.clear()

    if run_flag:
        run_flag.value = False

    for t in threads:
        t.join(timeout=0.1)

    for p in processes:
        p.terminate()

    # przywracamy standardowe locki
    locks = [threading.Lock() for _ in range(N)]


# -----------------------------
# MAIN LOOP — animacja
# -----------------------------

running = True
while running:

    # obsługa klawiatury
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            stop_all()

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False
                stop_all()

            if event.key == pygame.K_1:
                stop_all()
                threads = []
                processes = []
                states = [0] * N
                start_sequential()

            if event.key == pygame.K_2:
                stop_all()
                threads = []
                processes = []
                states = [0] * N
                start_threading()

            if event.key == pygame.K_3:
                stop_all()
                threads = []
                processes = []
                start_multiprocessing()

            if event.key == pygame.K_SPACE:
                if running_sim:
                    stop_all()
                else:
                    if mode == "seq":
                        start_sequential()
                    elif mode == "thread":
                        start_threading()
                    elif mode == "mp":
                        start_multiprocessing()

    # rysowanie
    screen.fill((30, 30, 30))

    for i, (x, y) in enumerate(positions):
        pygame.draw.circle(screen, COLORS[states[i]], (x, y), 60)
        txt = font.render(f"F{i}", True, (5, 5, 5))
        screen.blit(txt, (x - 20, y - 15))

    mode_txt = font.render(f"Tryb: {mode.upper()}", True, (200, 200, 200))
    screen.blit(mode_txt, (20, 20))
    pygame.draw.rect(screen, (50,50,50), (15, 50, 400, 120), border_radius=10)

    legend1 = font.render("Zielony Je — sekcja krytyczna", True, (0,200,70))
    legend2 = font.render("Zółty Czeka — brak zasobów", True, (255,180,0))
    legend3 = font.render("Szary Myśli — brak aktywności", True, (170,170,170))

    screen.blit(legend1, (20, 60))
    screen.blit(legend2, (20, 95))
    screen.blit(legend3, (20, 130))

    ctrl_txt = font.render("1=SEQ  2=THREADS  3=MP   SPACE=START/STOP   ESC=EXIT", True, (200, 200, 200))
    screen.blit(ctrl_txt, (20, 860))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
