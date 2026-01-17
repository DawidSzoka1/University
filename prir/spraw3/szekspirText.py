import torch
import torch.nn as nn
import requests
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# --- 1. PRZYGOTOWANIE DANYCH ---
print("Pobieranie dzieł Szekspira...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f"Liczba unikalnych znaków: {vocab_size}")

# Funkcja pomocnicza: pobiera losowy fragment tekstu
chunk_len = 200
def get_random_batch(batch_size=100, device='cpu'):
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

# --- 2. MODEL LSTM ---
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.rnn(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))

# --- 3. TEST WYDAJNOŚCI (BENCHMARK) ---
def run_benchmark(device_name, num_batches=20):
    device = torch.device(device_name)
    # Tworzymy model testowy
    model = RNN(vocab_size, 256, vocab_size, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print(f"--- Start testu na: {device_name.upper()} ({num_batches} wsadów) ---")
    start_time = time.time()

    for _ in range(num_batches):
        inp, target = get_random_batch(64, device) # Batch 64
        hidden = model.init_hidden(64, device)
        model.zero_grad()
        output, _ = model(inp, hidden)
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

    duration = time.time() - start_time
    print(f"Zakończono {device_name}. Czas: {duration:.4f} s")
    return duration

# Uruchamiamy testy
print("\n=== CZĘŚĆ 1: POMIAR AKCELERACJI ===")
# Mierzymy CPU
time_cpu = run_benchmark('cpu', num_batches=15)

# Mierzymy GPU
if torch.cuda.is_available():
    time_gpu = run_benchmark('cuda', num_batches=15)
    speedup = time_cpu / time_gpu
    print(f"\n>>> PRZYSPIESZENIE GPU: {speedup:.2f}x <<<")

    # Rysowanie wykresu
    plt.figure(figsize=(10, 5))
    plt.bar(['CPU', 'GPU'], [time_cpu, time_gpu], color=['red', 'green'])
    plt.title(f'Porównanie czasu treningu (15 wsadów)\nPrzyspieszenie: {speedup:.2f}x')
    plt.ylabel('Czas (sekundy)')
    plt.show()
else:
    print("Błąd: GPU niedostępne! Włącz GPU w ustawieniach Runtime.")
    time_gpu = time_cpu

# --- 4. TRENING WŁAŚCIWY (Dłuższy, na GPU) ---
print("\n=== CZĘŚĆ 2: TRENING MODELU DO CZATU ===")
print("Trenuję sieć, aby nauczyła się pisać (ok. 2000 iteracji)...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Większy model dla lepszej jakości
model = RNN(vocab_size, 512, vocab_size, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

start_train = time.time()
loss_history = []

for epoch in range(2001):
    inp, target = get_random_batch(100, device)
    hidden = model.init_hidden(100, device)

    model.zero_grad()
    output, _ = model(inp, hidden)
    loss = criterion(output.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoka {epoch}/2000 | Loss: {loss.item():.4f}")
        loss_history.append(loss.item())

print(f"Trening zakończony w {time.time()-start_train:.0f} sekund.")

# --- 5. CZAT ---
def generate_reply(model, start_sentence):
    model.eval()
    hidden = model.init_hidden(1, device)

    # Konwertujemy tekst usera na tensory
    inp = torch.tensor([char_to_int[c] for c in start_sentence if c in char_to_int]).unsqueeze(0).to(device)

    # Jeśli wpisano znaki spoza słownika, zwróć pusty string
    if inp.size(1) == 0: return "..."

    # "Wczytujemy" zdanie usera do pamięci LSTMa
    _, hidden = model(inp, hidden)
    inp = inp[:, -1].unsqueeze(1)

    predicted_text = ""
    for i in range(200):
        output, hidden = model(inp, hidden)

        # Temperatura (0.7 = bezpieczniej, 1.0 = losowo)
        output_dist = output.data.view(-1).div(0.7).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        char = int_to_char[top_i.item()]
        predicted_text += char

        if char == '\n' and len(predicted_text) > 10:
            break
        inp = torch.tensor([[top_i]], device=device)

    return predicted_text

print("\n" + "="*40)
print("INTERAKTYWNY SZEKSPIR")
print("Pisz po angielsku (np. 'The King', 'Where art thou').")
print("Wpisz 'exit', aby zakończyć.")
print("="*40)

while True:
    user_input = input("\nTY: ")
    if user_input.lower() == 'exit':
        break
    try:
        reply = generate_reply(model, user_input)
        print(f"SZEKSPIR: {reply.strip()}")
    except Exception as e:
        print("Szekspir: (Nie rozumiem tych znaków...)")