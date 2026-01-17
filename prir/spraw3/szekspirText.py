import torch
import torch.nn as nn
import requests
import time
import random
import matplotlib.pyplot as plt
import csv
import numpy as np
import os

# ==========================================
# 1. PRZYGOTOWANIE DANYCH (Szekspir)
# ==========================================
print("--- 1. Pobieranie dzieł Szekspira ---")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
try:
    text = requests.get(url).text
except:
    text = "To be or not to be, that is the question. " * 1000

chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f"Liczba znaków: {len(text)}, Unikalnych: {vocab_size}")

chunk_len = 200
def get_random_batch(batch_size=64, device='cpu'):
    input_tensor = torch.zeros(batch_size, chunk_len, dtype=torch.long).to(device)
    target_tensor = torch.zeros(batch_size, chunk_len, dtype=torch.long).to(device)
    for b in range(batch_size):
        start_index = random.randint(0, len(text) - chunk_len - 1)
        chunk = text[start_index : start_index + chunk_len + 1]
        input_data = [char_to_int[c] for c in chunk[:-1]]
        target_data = [char_to_int[c] for c in chunk[1:]]
        input_tensor[b] = torch.tensor(input_data)
        target_tensor[b] = torch.tensor(target_data)
    return input_tensor, target_tensor

# ==========================================
# 2. DEFINICJA MODELU LSTM
# ==========================================
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.rnn(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device))

# ==========================================
# 3. FAZA BENCHMARKU (Wydajność)
# ==========================================
def run_single_test(device_name, iterations):
    device = torch.device(device_name)
    model = RNN(vocab_size, 256, vocab_size).to(device) # Mniejszy model do testów
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    # Warmup
    if device_name == 'cuda':
        inp, tgt = get_random_batch(1, device)
        h = model.init_hidden(1, device)
        model(inp, h)

    start = time.time()
    for _ in range(iterations):
        inp, target = get_random_batch(64, device)
        hidden = model.init_hidden(64, device)
        model.zero_grad()
        output, _ = model(inp, hidden)
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

    if device_name == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start

print("\n--- 2. Rozpoczynam BENCHMARK (10-100 iteracji) ---")
ITERATION_LIST = [10, 20, 30, 40, 50, 60, 100]
results = []
gpu_avail = torch.cuda.is_available()

print(f"{'Iteracje':<10} | {'CPU (s)':<10} | {'GPU (s)':<10} | {'Speedup':<10}")
print("-" * 50)

for n in ITERATION_LIST:
    t_cpu = run_single_test('cpu', n)

    if gpu_avail:
        t_gpu = run_single_test('cuda', n)
        speedup = t_cpu / t_gpu
    else:
        t_gpu = t_cpu
        speedup = 1.0

    results.append({'iters': n, 'cpu': t_cpu, 'gpu': t_gpu, 'speedup': speedup})
    print(f"{n:<10} | {t_cpu:<10.4f} | {t_gpu:<10.4f} | {speedup:.2f}x")

# Zapis CSV i PNG
csv_file = "wyniki_benchmarku.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['iters', 'cpu', 'gpu', 'speedup'])
    writer.writeheader()
    writer.writerows(results)

# Rysowanie
iters = [r['iters'] for r in results]
cpu_times = [r['cpu'] for r in results]
gpu_times = [r['gpu'] for r in results]
speedups = [r['speedup'] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(iters, cpu_times, 'o-', color='red', label='CPU')
if gpu_avail: ax1.plot(iters, gpu_times, 's-', color='green', label='GPU')
ax1.set_title("Czas wykonania")
ax1.set_xlabel("Iteracje"); ax1.set_ylabel("Sekundy"); ax1.legend(); ax1.grid(True)

ax2.bar([str(i) for i in iters], speedups, color='purple', alpha=0.7)
ax2.set_title("Przyspieszenie (Speedup)")
ax2.set_xlabel("Iteracje"); ax2.set_ylabel("Krotność (x)"); ax2.grid(axis='y', linestyle='--')
plt.savefig("wykres_benchmark.png")
print("\n-> Wyniki zapisano: wyniki_benchmarku.csv oraz wykres_benchmark.png")


# ==========================================
# 4. TRENING DO CZATU (Jakość)
# ==========================================
print("\n--- 3. Trenowanie modelu do rozmowy ---")
print("(To zajmie chwilę, ale Szekspir musi się nauczyć mówić...)")

# Wybieramy najlepsze urządzenie do treningu
device_chat = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Większy model i więcej epok dla lepszej jakości tekstu
chat_model = RNN(vocab_size, 512, vocab_size).to(device_chat)
optimizer = torch.optim.Adam(chat_model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

start_train = time.time()
EPOCHS = 2000 # 2000 epok, żeby tekst miał sens

for epoch in range(1, EPOCHS + 1):
    inp, target = get_random_batch(100, device_chat)
    hidden = chat_model.init_hidden(100, device_chat)

    chat_model.zero_grad()
    output, _ = chat_model(inp, hidden)
    loss = criterion(output.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Postęp: {epoch}/{EPOCHS} | Loss: {loss.item():.4f}")

print(f"Trening zakończony w {time.time()-start_train:.1f}s.")

# ==========================================
# 5. INTERAKTYWNY SZEKSPIR
# ==========================================
def generate_reply(model, start_sentence, temp=0.8):
    model.eval()
    hidden = model.init_hidden(1, device_chat)

    # Przygotowanie wejścia
    inp_list = [char_to_int[c] for c in start_sentence if c in char_to_int]
    if not inp_list: return "..."
    inp = torch.tensor(inp_list).unsqueeze(0).to(device_chat)

    # "Wstępne" przepuszczenie przez sieć
    _, hidden = model(inp, hidden)
    inp = inp[:, -1].unsqueeze(1)

    predicted_text = ""

    # Generowanie 200 znaków
    for i in range(200):
        output, hidden = model(inp, hidden)

        # Sampling z temperaturą
        output_dist = output.data.view(-1).div(temp).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        char = int_to_char[top_i.item()]
        predicted_text += char

        # Koniec zdania po kropce lub nowej linii (jeśli długie)
        if (char == '\n' or char == '.') and len(predicted_text) > 20:
            break

        inp = torch.tensor([[top_i]], device=device_chat)

    return predicted_text

print("\n" + "="*50)
print(" INTERAKTYWNY SZEKSPIR (Po testach) ")
print(" Benchmark zakończony. Teraz możesz porozmawiać.")
print(" Pisz po angielsku. Wpisz 'exit', aby wyjść.")
print("="*50)

while True:
    user_input = input("\nTY: ")
    if user_input.lower() in ['exit', 'quit', 'wyjscie']:
        print("Szekspir: Adieu! Parting is such sweet sorrow...")
        break

    try:
        reply = generate_reply(chat_model, user_input)
        print(f"SZEKSPIR: {reply.strip()}")
    except Exception as e:
        print(f"Błąd: {e}")