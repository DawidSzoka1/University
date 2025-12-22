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

print(f"Wczytano {len(df_scene)} punktów geometrii (będzie gęsto!).")

# ==========================================
# 1. WYKRESY WYDAJNOŚCI (Matplotlib)
# ==========================================
# Tworzymy 4 wykresy (2x2)
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Analiza wydajności Ray Tracingu (CUDA vs CPU)', fontsize=16)

# --- A. Czas vs Rozdzielczość ---
valid_cpu = df_res[df_res['cpu_time_ms'] > 0]
axs[0, 0].plot(valid_cpu['pixels'], valid_cpu['cpu_time_ms'], 'r-o', label='CPU')
axs[0, 0].plot(df_res['pixels'], df_res['gpu_time_ms'], 'b-s', label='GPU')
axs[0, 0].set_title('Czas wykonania vs Rozdzielczość')
axs[0, 0].set_ylabel('Czas [ms] (Log)')
axs[0, 0].set_xlabel('Liczba pikseli')
axs[0, 0].set_yscale('log')
axs[0, 0].grid(True, linestyle='--')
axs[0, 0].legend()

# --- B. Przyspieszenie (Speedup) ---
speedup = valid_cpu['cpu_time_ms'].values / df_res.loc[valid_cpu.index, 'gpu_time_ms'].values
labels_speed = valid_cpu['label'].values
bars = axs[0, 1].bar(labels_speed, speedup, color='green')
axs[0, 1].set_title('Przyspieszenie GPU (Ile razy szybciej)')
axs[0, 1].set_ylabel('Krotność (x)')
axs[0, 1].tick_params(axis='x', rotation=45)
for bar in bars:
    axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{bar.get_height():.0f}x', ha='center', va='bottom', fontsize=9)

# --- C. Czas vs Liczba Wątków w Bloku (TEGO CHCIAŁEŚ) ---
# Oś X to liczba wątków (16, 64, 256, 1024), Oś Y to czas
axs[1, 0].plot(df_blocks['total_threads'], df_blocks['gpu_time_ms'], 'o-', color='orange', linewidth=2)
axs[1, 0].set_title('Czas wykonania vs Liczba wątków w bloku (4K)')
axs[1, 0].set_xlabel('Liczba wątków w bloku (Block Size^2)')
axs[1, 0].set_ylabel('Czas [ms]')
axs[1, 0].set_xticks(df_blocks['total_threads'])
axs[1, 0].grid(True)
for i, row in df_blocks.iterrows():
    axs[1, 0].text(row['total_threads'], row['gpu_time_ms'],
                   f"{int(row['gpu_time_ms'])}ms", ha='center', va='bottom')

# --- D. Tabelka z wynikami (pomocnicza) ---
axs[1, 1].axis('off')
table_data = df_res[['label', 'gpu_time_ms']].copy()
table_data['gpu_time_ms'] = table_data['gpu_time_ms'].apply(lambda x: f"{x:.2f}")
table = axs[1, 1].table(cellText=table_data.values, colLabels=["Rozdzielczość", "Czas GPU (ms)"], loc='center')
table.scale(1, 1.2)
axs[1, 1].set_title('Szczegółowe wyniki GPU')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_2d = 'wykresy_wydajnosc.png'
plt.savefig(output_2d, dpi=300) # dpi=300 oznacza wysoką jakość
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