import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import gradio as gr
import time
import os
import matplotlib.pyplot as plt # Do wykresów
import numpy as np

# --- KONFIGURACJA MODELU (LOGIKA GATYSA) ---
class VGG_NST(nn.Module):
    def __init__(self):
        super(VGG_NST, self).__init__()
        # Wybieramy konkretne warstwy do ekstrakcji stylu i treści
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = vgg19(weights=VGG19_Weights.DEFAULT).features[:29] # do warstwy conv5_1

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image, size=400, device="cpu"):
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return loader(image).unsqueeze(0).to(device)

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# --- GŁÓWNA FUNKCJA TRANSFORMACJI ---
def run_style_transfer(content_img, style_img, device_choice, steps=100):
    # Wybór urządzenia
    if device_choice == "GPU (CUDA)" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Inicjalizacja obrazów i modelu
    content_p = load_image(content_img, device=device)
    style_p = load_image(style_img, device=device)
    generated = content_p.clone().requires_grad_(True)

    model = VGG_NST().to(device).eval()
    optimizer = optim.Adam([generated], lr=0.02)

    # Wagi dla stylu i treści
    style_weight = 1000000
    content_weight = 1

    start_time = time.time()
    for step in range(steps):
        generated_features = model(generated)
        content_features = model(content_p)
        style_features = model(style_p)

        style_loss = content_loss = 0

        for gen_f, con_f, sty_f in zip(generated_features, content_features, style_features):
            # Content Loss
            content_loss += torch.mean((gen_f - con_f)**2)

            # Style Loss
            G = gram_matrix(gen_f)
            A = gram_matrix(sty_f)
            style_loss += torch.mean((G - A)**2)

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    duration = time.time() - start_time

    # Postprocessing
    out = generated.to("cpu").clone().detach().squeeze(0)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    out = inv_normalize(out)
    out = torch.clamp(out, 0, 1)

    # Zwracamy obrazek oraz czas jako liczbę (float) dla łatwiejszego benchmarku
    return transforms.ToPILImage()(out), duration

# Wrapper dla UI (pojedyncza transformacja), żeby zwracał string czasu
def run_style_transfer_ui(c, s, d, steps=100):
    img, time_val = run_style_transfer(c, s, d, steps)
    return img, f"{time_val:.2f} s"

# --- FUNKCJA BENCHMARKOWA (NOWOŚĆ) ---
def compare_performance(content_img, style_img):
    if not torch.cuda.is_available():
        return None, "Brak GPU (CUDA) w systemie. Nie można przeprowadzić porównania."

    steps_test = 40 # Krótki test, żeby nie czekać zbyt długo

    # 1. Test na CPU
    # print("Testowanie CPU...")
    _, time_cpu = run_style_transfer(content_img, style_img, "CPU", steps=steps_test)

    # 2. Test na GPU
    # print("Testowanie GPU...")
    _, time_gpu = run_style_transfer(content_img, style_img, "GPU (CUDA)", steps=steps_test)

    # 3. Rysowanie wykresu
    labels = ['CPU', 'GPU (CUDA)']
    times = [time_cpu, time_gpu]
    colors = ['#FF6B6B', '#4ECDC4'] # Czerwony i turkusowy

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, times, color=colors)

    ax.set_ylabel('Czas wykonania (sekundy)')
    ax.set_title(f'Porównanie czasu (dla {steps_test} kroków)')

    # Dodanie wartości nad słupkami
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    speedup = time_cpu / time_gpu

    result_text = (
        f"Czas CPU: {time_cpu:.2f} s\n"
        f"Czas GPU: {time_gpu:.2f} s\n"
        f"PRZYSPIESZENIE: {speedup:.2f}x"
    )

    return fig, result_text

# --- FUNKCJA BATCH PROCESSING ---
def batch_process(style_img, folder_path):
    if not os.path.exists(folder_path): return "Folder nie istnieje."
    out_dir = os.path.join(folder_path, "styled_output")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for f in files:
        img = Image.open(os.path.join(folder_path, f)).convert('RGB')
        # Używamy GPU domyślnie dla batcha
        res, _ = run_style_transfer(img, style_img, "GPU (CUDA)", steps=50)
        res.save(os.path.join(out_dir, f"art_{f}"))
    return f"Gotowe! Przetworzono {len(files)} zdjęć."

# --- INTERFEJS GRADIO ---
with gr.Blocks(title="Pro NST Tool") as demo:
    gr.Markdown("# Profesjonalny Transfer Stylu + Benchmark")

    with gr.Tabs():
        # TAB 1: Pojedyncze zdjęcie
        with gr.Tab("Pojedyncza Transformacja"):
            with gr.Row():
                with gr.Column():
                    c_in = gr.Image(label="Treść", type="pil")
                    s_in = gr.Image(label="Styl", type="pil")
                    dev = gr.Radio(["CPU", "GPU (CUDA)"], label="Sprzęt", value="GPU (CUDA)")
                    btn = gr.Button("GENERUJ DZIEŁO")
                with gr.Column():
                    res_out = gr.Image(label="Wynik")
                    t_out = gr.Textbox(label="Czas")
            btn.click(run_style_transfer_ui, [c_in, s_in, dev], [res_out, t_out])

        # TAB 2: Benchmark (NOWOŚĆ)
        with gr.Tab("Benchmark CPU vs GPU"):
            gr.Markdown("Test wykonuje 40 kroków optymalizacji na obu urządzeniach, aby porównać wydajność.")
            with gr.Row():
                with gr.Column():
                    b_c_in = gr.Image(label="Treść do testu", type="pil")
                    b_s_in = gr.Image(label="Styl do testu", type="pil")
                    bench_btn = gr.Button("URUCHOM TEST PORÓWNAWCZY")
                with gr.Column():
                    plot_out = gr.Plot(label="Wykres wydajności")
                    stats_out = gr.Textbox(label="Szczegóły", lines=4)

            bench_btn.click(compare_performance, [b_c_in, b_s_in], [plot_out, stats_out])

        # TAB 3: Folder
        with gr.Tab("Folder (Batch)"):
            s_batch = gr.Image(label="Styl", type="pil")
            f_path = gr.Textbox(label="Ścieżka do folderu")
            b_btn = gr.Button("PROCESUJ FOLDER (GPU)")
            b_status = gr.Textbox(label="Status")
            b_btn.click(batch_process, [s_batch, f_path], b_status)

demo.launch()