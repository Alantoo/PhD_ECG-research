# pip install numpy scipy matplotlib
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, resample_poly, detrend


# ----------------------------- базові функції -----------------------------
def butter_bandpass(x, fs, low=0.5, high=40.0, order=4):
    sos = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def decimate_signal(x, fs, factor, method="butter", order=8):
    if factor <= 1:
        return x, fs
    if method == "butter":
        fc = 0.45 * (fs / factor)
        sos = butter(order, fc / (fs / 2), btype="low", output="sos")
        xf = sosfiltfilt(sos, x)
        y = xf[::factor]
    elif method == "poly":
        y = resample_poly(x, up=1, down=factor)
    else:
        raise ValueError("method must be 'butter' or 'poly'")
    return y, fs / factor


# ----------------------------- синтетичний сигнал -----------------------------
def make_synthetic_ecg(fs=500, dur=6.0, hr_hz=1.2, noise_std=0.03):
    t = np.arange(0, dur, 1 / fs)
    tb = np.linspace(0, 1.0, int(fs * 1.0), endpoint=False)

    def gauss(mu, sigma, amp=1.0): return amp * np.exp(-0.5 * ((tb - mu) / sigma) ** 2)

    template = (gauss(0.18, 0.03, 0.12) - gauss(0.48, 0.012, 0.05)
                + gauss(0.50, 0.009, 1.1) - gauss(0.52, 0.013, 0.23)
                + gauss(0.78, 0.07, 0.33))
    spikes = np.zeros_like(t)
    spikes[(np.arange(0, dur, 1 / hr_hz) * fs).astype(int)] = 1.0
    ecg_clean = np.convolve(spikes, template, mode="same")
    baseline = 0.25 * np.sin(2 * np.pi * 0.25 * t)
    noise = noise_std * np.random.randn(len(t))
    return t, ecg_clean + baseline + noise


def apply_detrend(x):
    return detrend(x, type="linear")


def defaults_process_ecg_pipeline(x, fs=500, factor=4):
    y, fs_new = process_ecg_pipeline(x, fs, factor)
    t_dec = np.arange(len(y)) / fs_new
    return y, t_dec


def defaults_decimate_signal(x, fs=500, factor=4):
    y, fs_new = decimate_signal(x, fs, factor)
    t_dec = np.arange(len(y)) / fs_new
    return y, t_dec


# ----------------------------- повний пайплайн -----------------------------
def process_ecg_pipeline(x, fs, factor=4):
    # 1. усунення тренду
    x_detr = apply_detrend(x)
    # 2. смуговий фільтр 0.5–40 Гц
    x_band = butter_bandpass(x_detr, fs, 0.5, 40.0)
    # 3. децимація
    x_dec, fs_new = decimate_signal(x_band, fs, factor=factor, method="butter", order=8)
    # 4. нормалізація
    # x_norm = (x_detr - np.mean(x_detr)) / np.std(x_detr)
    return x_dec, fs_new


# ----------------------------- демо -----------------------------
if __name__ == "__main__":
    fs = 500
    # t, x = make_synthetic_ecg(fs=fs)
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
    x = np.transpose(rawSignal).tolist()[1]
    t = np.transpose(rawSignal).tolist()[0]

    # застосування пайплайну
    y, fs_new = process_ecg_pipeline(x, fs, factor=4)
    t_dec = np.arange(len(y)) / fs_new

    # Візуалізація
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="Original (500 Hz)", alpha=0.6)
    plt.plot(t_dec, y, label="Processed (125 Hz)", linewidth=1.2)
    plt.title("ECG Pipeline: detrend → bandpass → decimate → normalize")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (normalized)")
    plt.legend()
    plt.tight_layout()

    # Показати
    plt.show()

    # Зберегти
    # plt.savefig("ecg_pipeline_result.png", dpi=150, bbox_inches="tight")
    # print(f"Saved to ecg_pipeline_result.png (fs_new={fs_new} Hz)")
