# demo_detrend.py
# pip install numpy scipy matplotlib
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, medfilt, savgol_filter

# ----------------------------- базові методи -----------------------------
def detrend_hpf(x, fs, fc=0.5, order=4):
    sos = butter(order, fc/(fs/2), btype="highpass", output="sos")
    y = sosfiltfilt(sos, x)
    return y, x - y  # (detrended, baseline)

def detrend_median(x, fs, w1_sec=0.22, w2_sec=0.9):
    w1 = int(round(w1_sec * fs)) | 1
    w2 = int(round(w2_sec * fs)) | 1
    w1 = max(3, w1 | 1)
    w2 = max(w1 + 2, w2 | 1)
    m1 = medfilt(x, kernel_size=w1)
    baseline = medfilt(m1, kernel_size=w2)
    return x - baseline, baseline

def detrend_savgol(x, fs, win_sec=0.9, poly=3):
    w = int(round(win_sec * fs)) | 1
    baseline = savgol_filter(x, window_length=max(5, w), polyorder=poly)
    return x - baseline, baseline

def _als_baseline(y, lam=3e5, p=0.01, niter=6):
    """Asymmetric Least Squares (Eilers). Щільна реалізація (простий np.linalg.solve)."""
    y = np.asarray(y, float)
    L = len(y)
    # Матриця другої різниці (L-2) x L
    D = np.zeros((L-2, L))
    idx = np.arange(L-2)
    D[idx, idx]     = 1.0
    D[idx, idx + 1] = -2.0
    D[idx, idx + 2] = 1.0
    DTD = D.T @ D  # L x L

    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        z = np.linalg.solve(W + lam * DTD, W @ y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z

def detrend_als(x, fs, lam=3e5, p=0.01, niter=6):
    baseline = _als_baseline(x, lam=lam, p=p, niter=niter)
    return x - baseline, baseline

# ----------------------------- синтетичні дані -----------------------------
def make_synthetic_ecg(fs=200, dur=6.0, hr_hz=1.1, noise_std=0.04):
    t = np.arange(0, dur, 1/fs)
    beat_len = int(fs * 1.0)
    tb = np.linspace(0, 1.0, beat_len, endpoint=False)

    def gauss(mu, sigma, amp=1.0):
        return amp * np.exp(-0.5*((tb-mu)/sigma)**2)

    template = (gauss(0.18, 0.03, 0.12) - gauss(0.48, 0.012, 0.05) +
                gauss(0.50, 0.009, 1.2) - gauss(0.52, 0.013, 0.25) +
                gauss(0.78, 0.07, 0.35))

    spike_pos = np.arange(0, dur, 1/hr_hz)
    spikes = np.zeros_like(t)
    spikes[(spike_pos*fs).astype(int)] = 1.0
    ecg_clean = np.convolve(spikes, template, mode="same")

    baseline_wander = 0.3 * np.sin(2*np.pi*0.28*t) + 0.0015*(t - t.mean())**2
    noise = noise_std * np.random.randn(len(t))
    ecg = ecg_clean + baseline_wander + noise
    return t, ecg, baseline_wander

# ----------------------------- демо -----------------------------
if __name__ == "__main__":
    fs = 500
    # t, ecg, baseline_true = make_synthetic_ecg(fs=fs)

    methods = {
        "HPF": lambda x: detrend_hpf(x, fs, fc=0.5, order=4),
        "Median": lambda x: detrend_median(x, fs, w1_sec=0.22, w2_sec=0.9),
        "Savitzky-Golay": lambda x: detrend_savgol(x, fs, win_sec=0.9, poly=3),
        "ALS": lambda x: detrend_als(x, fs, lam=3e5, p=0.01, niter=6),
    }

    file_path = 'demo_data/Test - Main signal.json'  # Replace with the actual path to your file
    file_content = ''
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    rawSignal = json.loads(file_content)
    ecg = np.transpose(rawSignal).tolist()[1]
    t = np.transpose(rawSignal).tolist()[0]

    # 0) оригінал + змодельований дрейф
    plt.figure()
    plt.plot(t, ecg, label="Original (with drift)")
    # plt.plot(t, baseline_true, label="Underlying drift (simulated)")
    plt.title("Synthetic ECG: original vs. simulated baseline")
    plt.xlabel("Time, s"); plt.ylabel("Amplitude"); plt.legend()
    plt.savefig("demo_detrend/00_original_vs_baseline.png", dpi=140, bbox_inches="tight")
    plt.close()

    os.makedirs("demo_detrend", exist_ok=True)
    # 1..N) по кожному методу: baseline і detrended
    for name, fn in methods.items():
        y, base = fn(ecg)

        plt.figure()
        plt.plot(t, ecg, label="Original")
        plt.plot(t, base, label="Estimated baseline")
        plt.title(f"{name}: baseline estimate")
        plt.xlabel("Time, s"); plt.ylabel("Amplitude"); plt.legend()
        plt.savefig(f"demo_detrend/{name}_baseline.png".replace(" ", "_"), dpi=140, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(t, y, label="Detrended")
        plt.title(f"{name}: detrended signal")
        plt.xlabel("Time, s"); plt.ylabel("Amplitude"); plt.legend()
        plt.savefig(f"demo_detrend/{name}_detrended.png".replace(" ", "_"), dpi=140, bbox_inches="tight")
        plt.close()

    print("Saved: 00_original_vs_baseline.png and pairs *_baseline.png / *_detrended.png")
