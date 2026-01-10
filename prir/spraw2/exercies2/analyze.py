import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import os

# Sprawdzenie plików
required_files = ['benchmark_resolution.csv', 'benchmark_blocks.csv', 'scene_data.csv']
for f in required_files:
    if not os.path.exists(f):
        print(f"BŁĄD: Nie znaleziono pliku '{f}'. Uruchom program C++!")
        exit()

# Wczytanie danych
df_res = pd.read_csv('benchmark_resolution.csv')
df_blocks = pd.read_csv('benchmark_blocks.csv')
df_scene = pd.read_csv('scene_data.csv')

# ==========================================
# 1. WYKRESY WYDAJNOŚCI (Matplotlib)
# ==========================================
# Zmieniamy układ na 3x2, aby zmieścić dodatkowe analizy wątków
fig, axs = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle('Zaawansowana Analiza Wydajności Ray Tracingu - RTX 5080', fontsize=20)

# --- A. Czas vs Rozdzielczość (Log Scale) ---
valid_cpu = df_res[df_res['cpu_time_ms'] > 0]
axs[0, 0].plot(valid_cpu['pixels'], valid_cpu['cpu_time_ms'], 'r-o', label='CPU', linewidth=2)
axs[0, 0].plot(df_res['pixels'], df_res['gpu_time_ms'], 'b-s', label='GPU', linewidth=2)
axs[0, 0].set_title('Czas wykonania vs Rozdzielczość')
axs[0, 0].set_yscale('log')
axs[0, 0].grid(True, which="both", linestyle='--', alpha=0.5)
axs[0, 0].legend()

# --- B. Przyspieszenie (Speedup) ---
speedup = valid_cpu['cpu_time_ms'].values / df_res.loc[valid_cpu.index, 'gpu_time_ms'].values
axs[0, 1].bar(valid_cpu['label'], speedup, color='forestgreen')
axs[0, 1].set_title('Przyspieszenie GPU (Speedup Factor)')
axs[0, 1].set_ylabel('x-faster')
axs[0, 1].tick_params(axis='x', rotation=45)

# --- C. Czas vs Liczba Wątków w Bloku (Gęste dane) ---
# To pokazuje "sweet spot" dla Twojej karty
axs[1, 0].plot(df_blocks['total_threads'], df_blocks['gpu_time_ms'], 'o-', color='darkorange', linewidth=2.5)
# Zaznaczamy Warp (32 wątki) i standardowy blok (256 wątków) dla odniesienia
axs[1, 0].set_title('Wpływ rozmiaru bloku na czas (4K)')
axs[1, 0].set_xlabel('Całkowita liczba wątków w bloku')
axs[1, 0].set_ylabel('Czas [ms]')
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend()


# --- E. Porównanie różnych wielkości bloków (Wykres słupkowy) ---
axs[2, 0].bar(df_blocks['label'], df_blocks['gpu_time_ms'], color='teal')
axs[2, 0].set_title('Porównanie konkretnych konfiguracji bloków')
axs[2, 0].set_ylabel('Czas [ms]')
for i, val in enumerate(df_blocks['gpu_time_ms']):
    axs[2, 0].text(i, val, f"{val:.1f}", ha='center', va='bottom')

# --- F. Tabela wyników ---
axs[2, 1].axis('off')
table_data = df_blocks[['label', 'total_threads', 'gpu_time_ms']].copy()
table = axs[2, 1].table(cellText=table_data.values, colLabels=["Blok", "Wątki", "Czas GPU (ms)"], loc='center')
table.scale(1, 1.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('analiza_watkow_gpu.png', dpi=300)
plt.show()

# ==========================================
# 2. INTERAKTYWNA SCENA 3D (BARDZO GĘSTA)
# ==========================================
print("Generowanie interaktywnej sceny 3D z gęstej geometrii...")

fig_3d = go.Figure(data=[go.Scatter3d(
    x=df_scene['x'],
    y=df_scene['y'],
    z=df_scene['z'],
    mode='markers',
    marker=dict(
        size=3, # Małe punkciki, bo jest ich dużo
        color=df_scene[['r', 'g', 'b']].values,
        opacity=0.9
    )
)])

fig_3d.update_layout(
    title=f"Wizualizacja Inicjałów DS - {len(df_scene)} punktów (Wysoka Jakość)",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='black',
        aspectmode='data'
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)

fig_3d.show()