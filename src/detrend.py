import numpy as np
from scipy.signal import butter, sosfiltfilt, medfilt, savgol_filter, detrend as sp_detrend

def detrend_ecg(x, fs, method="hpf", **kw):
    """
    x: 1D масив сигналу
    fs: частота дискретизації (Гц)
    method: 'hpf' | 'median' | 'savgol' | 'poly' | 'als' | 'linear'
    Повертає: y (сигнал без тренду), baseline (оцінений тренд)
    """
    x = np.asarray(x)

    if method == "hpf":
        # Параметри: частота зрізу fc (Гц), порядок n
        fc = kw.get("fc", 0.5)
        n = kw.get("order", 4)
        sos = butter(n, fc/(fs/2), btype="highpass", output="sos")
        y = sosfiltfilt(sos, x)
        baseline = x - y
        return y, baseline

    elif method == "median":
        # Вікна у секундах конвертуємо у семпли
        w1 = int(round(kw.get("w1_sec", 0.2) * fs)) | 1  # 0.2 c → прибрати QRS
        w2 = int(round(kw.get("w2_sec", 0.6) * fs)) | 1  # 0.6 c → базова лінія
        # Запобігти занадто малим/парним вікнам
        w1 = max(3, w1 | 1); w2 = max(w1+2, w2 | 1)
        # 2-етапний медіан
        m1 = medfilt(x, kernel_size=w1)
        baseline = medfilt(m1, kernel_size=w2)
        y = x - baseline
        return y, baseline

    elif method == "savgol":
        # Вікно ~ 0.6–1.2 c, полінои 2–3
        w = int(round(kw.get("win_sec", 0.8) * fs)) | 1
        poly = kw.get("poly", 3)
        baseline = savgol_filter(x, window_length=max(5, w), polyorder=poly)
        y = x - baseline
        return y, baseline

    elif method == "poly":
        # Поліноміальна регресія на нормованій осі часу
        deg = kw.get("deg", 3)
        t = np.linspace(-1, 1, len(x))
        coef = np.polyfit(t, x, deg)
        baseline = np.polyval(coef, t)
        y = x - baseline
        return y, baseline

    elif method == "linear":
        # Швидкий лінійний detrend (SciPy)
        y = sp_detrend(x, type="linear")
        baseline = x - y
        return y, baseline

    elif method == "als":
        # Asymmetric Least Squares baseline (Eilers, 2004)
        lam = kw.get("lam", 1e6)   # згладжування (↑ — плавніша база)
        p   = kw.get("p", 0.01)    # асиметрія (0<p<1), для базової лінії типово 0.01–0.1
        baseline = _als_baseline(x, lam=lam, p=p, niter=kw.get("niter", 10))
        y = x - baseline
        return y, baseline

    else:
        raise ValueError("Unknown method")

def _als_baseline(y, lam=1e6, p=0.01, niter=10):
    """ALS baseline (без залежностей поза NumPy)."""
    y = np.asarray(y, float)
    L = len(y)
    D = np.diff(np.eye(L), 2)  # матриця другої похідної
    DTD = D.T @ D
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        # (W + lam*D^T D) z = W y
        z = np.linalg.solve(W + lam * DTD, W @ y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z
