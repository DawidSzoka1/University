import pywt
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as signal
import pandas as pd
import wfdb
import os

# ---------------------------------------------------------
# FUNKCJE WCZYTYWANIA SYGNAŁÓW Z .dat
# ---------------------------------------------------------

def load_signal(path):

    base = path.replace(".dat", "")

    # 1) WFDB
    if os.path.exists(base + ".hea"):
        rec = wfdb.rdrecord(base)
        sig = rec.p_signal
        if sig.ndim > 1:
            sig = sig[:,0]
        return sig, rec.fs, "WFDB"

    # 2) ASCII
    try:
        txt = np.loadtxt(path)
        if txt.ndim == 2 and txt.shape[1] >= 2:
            return txt[:,1], 360.0, "ASCII 2 kolumny"
        if txt.ndim == 1:
            return txt, 360.0, "ASCII 1 kolumna"
    except:
        pass

    # 3) BINARNY int16
    try:
        raw = np.fromfile(path, dtype=np.int16)
        return raw.astype(float), 360.0, "BIN int16"
    except:
        pass

    raise ValueError("Nie mogę odczytać pliku: " + path)


# ---------------------------------------------------------
# WCZYTANIE DWÓCH PLIKÓW
# ---------------------------------------------------------

sig_h, fs_h, src1 = load_signal("zdrowy.dat")
sig_d, fs_d, src2 = load_signal("bol.dat")

print("Wczytano zdrowy.dat →", src1, "  fs =", fs_h)
print("Wczytano bol.dat →", src2, "  fs =", fs_d)

fs = fs_h

L = min(len(sig_h), len(sig_d))
sig_h = sig_h[:L]
sig_d = sig_d[:L]

t = np.arange(L) / fs
print(t)

# ---------------------------------------------------------
# PSD (Welch)
# ---------------------------------------------------------
f_h, Pxx_h = signal.welch(sig_h, fs=fs, nperseg=2048)
f_d, Pxx_d = signal.welch(sig_d, fs=fs, nperseg=2048)

plt.figure(figsize=(8,4))
plt.semilogy(f_h, Pxx_h)
plt.title("PSD – zdrowy")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD")
plt.xlim(0,60)
plt.savefig("psd_zdrowy.png")
plt.show()

plt.figure(figsize=(8,4))
plt.semilogy(f_d, Pxx_d)
plt.title("PSD – chory")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD")
plt.xlim(0,60)
plt.savefig("psd_chory.png")
plt.show()


# ---------------------------------------------------------
# SPEKTROGRAM
# ---------------------------------------------------------
fH, tH, sH = signal.spectrogram(sig_h, fs=fs, nperseg=256, noverlap=200)
plt.figure(figsize=(8,4))
plt.pcolormesh(tH, fH, 10*np.log10(sH), shading='auto')
plt.title("Spektrogram – zdrowy")
plt.ylim(0,60)
plt.colorbar()
plt.savefig("spectrogram_zdrowy.png")
plt.show()

fD, tD, sD = signal.spectrogram(sig_d, fs=fs, nperseg=256, noverlap=200)
plt.figure(figsize=(8,4))
plt.pcolormesh(tD, fD, 10*np.log10(sD), shading='auto')
plt.title("Spektrogram – chory")
plt.ylim(0,60)
plt.colorbar()
plt.savefig("spectrogram_chory.png")
plt.show()


# ---------------------------------------------------------
# METRYKI
# ---------------------------------------------------------
def metrics(x, fs):
    peaks, _ = signal.find_peaks(x, distance=int(0.4*fs), height=np.percentile(x,75))
    rr = np.diff(peaks)/fs
    hr = 60/np.mean(rr) if len(rr) > 1 else np.nan
    return {
        "n_peaks": len(peaks),
        "HR [BPM]": np.round(hr, 2),
        "czas [s]": len(x)/fs
    }

df = pd.DataFrame([metrics(sig_h, fs), metrics(sig_d, fs)], index=["zdrowy","chory"])
print(df)


# PORÓWNANIE PSD NA JEDNYM WYKRESIE
plt.figure(figsize=(10,5))
plt.semilogy(f_h, Pxx_h, label="Zdrowy")
plt.semilogy(f_d, Pxx_d, label="Chory")
plt.title("Porównanie widma mocy – zdrowy vs chory")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("PSD (skala log)")
plt.xlim(0, 60)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare.png")
plt.show()

def fourier_dominant_freqs(signal, fs, n=5):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    spectrum = np.abs(np.fft.rfft(signal))

    idx = np.argsort(spectrum)[-n:]  # n największych
    idx = idx[np.argsort(freqs[idx])]  # posortowane rosnąco

    return freqs[idx], spectrum[idx]

# --- 4. ANALIZA FALKOWA (CWT) --------------------------------------------------
def wavelet_freqs(signal, fs, wavelet='morl', n=5):
    widths = np.arange(1, 256)
    cwt_matrix, _ = pywt.cwt(signal, widths, wavelet, sampling_period=1/fs)

    power = np.abs(cwt_matrix)**2
    avg_power = power.mean(axis=1)

    # przeliczenie skali na częstotliwości
    freqs = pywt.scale2frequency(wavelet, widths) * fs

    idx = np.argsort(avg_power)[-n:]
    idx = idx[np.argsort(freqs[idx])]

    return freqs[idx], avg_power[idx]

# ---. LICZENIE WYNIKÓW ------------------------------------------------------
f_fft_h, A_fft_h = fourier_dominant_freqs(sig_h, fs)
f_fft_d, A_fft_d = fourier_dominant_freqs(sig_d, fs)

f_cwt_h, A_cwt_h = wavelet_freqs(sig_h, fs)
f_cwt_d, A_cwt_d = wavelet_freqs(sig_d, fs)

# ---  TABELA Z WYNIKAMI -----------------------------------------------------
df = pd.DataFrame({
    "Metoda": ["FFT", "FFT", "FFT", "FFT", "FFT",
               "FFT", "FFT", "FFT", "FFT", "FFT"],

    "Sygnał": ["Zdrowy"]*5 + ["Chory"]*5,

    "Częstotliwość [Hz]": np.concatenate([f_fft_h, f_fft_d]),
    "Moc / Energia": np.concatenate([A_fft_h, A_fft_d])
})

print(df)

df = pd.DataFrame({
    "Metoda": ["CWT", "CWT", "CWT", "CWT", "CWT",
               "CWT", "CWT", "CWT", "CWT", "CWT"],

    "Sygnał": ["Zdrowy"]*5 + ["Chory"]*5,

    "Częstotliwość [Hz]": np.concatenate([f_cwt_h, f_cwt_d]),
    "Moc / Energia": np.concatenate([A_cwt_h, A_cwt_d])
})

print(df)
