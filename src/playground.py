import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import wfdb

sys.path.insert(0, os.path.dirname(__file__))

from preparedSignal import PreparedSignal

# ── signal to analyse ────────────────────────────────────────────────────────
PHYSIONET_BASE = "/Users/alantoo/Workspace/Edu/ecg_database/physionet.org/files"
DATABASE       = "pulse-transit-time-ppg/1.1.0"
DATAFILE       = "s1_sit"       # file stem, without extension
SIGNAL_IDX     = 0          # channel index inside the file
# ─────────────────────────────────────────────────────────────────────────────


def load_signal(base, database, datafile, signal_idx):
    path = f"{base}/{database}/{datafile}"
    signals, fields = wfdb.rdsamp(path)
    sampling_rate = fields["fs"]
    raw = signals[:, signal_idx]
    # same normalisation as ReadPhysionetFile
    raw = raw - np.mean(raw)
    if np.max(raw) > 1000:
        raw = raw / 1000.0
    return raw, sampling_rate


def unzip_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return xs, ys


def find_complete_beats(prepared, n=4):
    """Return indices of first n beats where all wave boundaries are non-NaN."""
    cols = [
        prepared.ECG_P_Onsets, prepared.ECG_P_Offsets,
        prepared.ECG_Q_Peaks,  prepared.ECG_S_Peaks,
        prepared.ECG_T_Onsets, prepared.ECG_T_Offsets,
        prepared.ECG_R_Peaks,
    ]
    beats = []
    for i in range(len(prepared.ECG_R_Peaks)):
        if all(not np.isnan(float(c.iloc[i])) for c in cols):
            beats.append(i)
        if len(beats) == n:
            break
    return beats


def plot_beat(ax, signal, fs, prepared, idx, title):
    p_on  = float(prepared.ECG_P_Onsets.iloc[idx])
    p_off = float(prepared.ECG_P_Offsets.iloc[idx])
    q     = float(prepared.ECG_Q_Peaks.iloc[idx])
    s     = float(prepared.ECG_S_Peaks.iloc[idx])
    t_on  = float(prepared.ECG_T_Onsets.iloc[idx])
    t_off = float(prepared.ECG_T_Offsets.iloc[idx])
    r     = float(prepared.ECG_R_Peaks.iloc[idx])

    def safe(series, i):
        if i < 0 or i >= len(series):
            return None
        v = float(series.iloc[i])
        return None if np.isnan(v) else v

    next_p_on = safe(prepared.ECG_P_Onsets, idx + 1)

    win_end = next_p_on if next_p_on is not None else t_off

    seg_start = max(0, int(p_on * fs))
    seg_end   = min(len(signal), int(win_end * fs))
    t = np.arange(seg_start, seg_end) / fs

    ax.plot(t, signal[seg_start:seg_end], color="#333333", linewidth=1.2)

    ax.axvspan(p_on,  p_off, color="#457b9d", alpha=0.3, label="P wave")
    ax.axvspan(p_off, q,     color="#aaaaaa", alpha=0.2, label="PR segment")
    ax.axvspan(q,     s,     color="#e63946", alpha=0.3, label="QRS")
    ax.axvspan(s,     t_on,  color="#aaaaaa", alpha=0.2, label="ST segment")
    ax.axvspan(t_on,  t_off, color="#2a9d8f", alpha=0.3, label="T wave")
    if next_p_on is not None:
        ax.axvspan(t_off, next_p_on, color="#e9c46a", alpha=0.2, label="TP segment")
    ax.axvline(r, color="#e63946", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def main():
    print(f"Loading {DATABASE}/{DATAFILE} channel {SIGNAL_IDX} …")
    signal, fs = load_signal(PHYSIONET_BASE, DATABASE, DATAFILE, SIGNAL_IDX)
    print(f"  samples={len(signal)}, fs={fs} Hz, duration={len(signal)/fs:.1f}s")

    print("Running PreparedSignal …")
    prepared = PreparedSignal(signal.tolist(), fs)

    print("Computing stats …")
    stats = prepared.get_stats()

    mean_points   = stats["mathematical_expectation"]
    rhythm_points = prepared.rhythm_points

    print(f"  mean points:   {len(mean_points)}")
    print(f"  rhythm points: {len(rhythm_points)}")
    print(f"  P wave beats:  {len(prepared.matrix_P_wave)}")
    print(f"  QRS beats:     {len(prepared.matrix_QRS)}")
    print(f"  T wave beats:  {len(prepared.matrix_T_wave)}")

    beat_indices = find_complete_beats(prepared, n=4)
    print(f"  complete beats found: {len(beat_indices)}")

    mx, my = unzip_points(mean_points)
    rx, ry = unzip_points(rhythm_points)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{DATABASE} / {DATAFILE}  [ch {SIGNAL_IDX}]", fontsize=13)

    # top row: 3 main plots
    ax_raw    = fig.add_subplot(3, 1, 1)
    ax_mean   = fig.add_subplot(3, 2, 3)
    ax_rhythm = fig.add_subplot(3, 2, 4)

    # bottom row: up to 4 beat segments
    beat_axes = [fig.add_subplot(3, 4, 9 + k) for k in range(4)]

    t = np.arange(len(signal)) / fs
    mask = t <= 10
    ax_raw.plot(t[mask], signal[mask], linewidth=0.8, color="#1f77b4")
    ax_raw.set_title("Raw signal (first 10 s)")
    ax_raw.set_xlabel("Time (s)")
    ax_raw.set_ylabel("Amplitude")
    ax_raw.grid(True, alpha=0.3)

    ax_mean.plot(mx, my, linewidth=1.2, color="#2ca02c")
    ax_mean.set_title("Mathematical expectation (mean beat shape)")
    ax_mean.set_xlabel("Sample position")
    ax_mean.set_ylabel("Amplitude")
    ax_mean.grid(True, alpha=0.3)

    ax_rhythm.plot(rx, ry, linewidth=1.0, color="#d62728", marker=".", markersize=3)
    ax_rhythm.set_title("Rhythm — interleaved inter-peak intervals")
    ax_rhythm.set_xlabel("Beat index")
    ax_rhythm.set_ylabel("Interval (s)")
    ax_rhythm.grid(True, alpha=0.3)

    for k, beat_idx in enumerate(beat_indices):
        plot_beat(beat_axes[k], signal, fs, prepared, beat_idx, f"Beat #{beat_idx + 1}")

    for k in range(len(beat_indices), 4):
        beat_axes[k].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
