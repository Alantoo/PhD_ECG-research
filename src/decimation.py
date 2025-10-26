import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


def decimate_signal(x, fs, factor, method="butter", order=6):
    """
    Зменшує частоту дискретизації сигналу, запобігаючи aliasing.

    Параметри
    ----------
    x : array_like
        Вхідний сигнал.
    fs : float
        Початкова частота дискретизації (Гц).
    factor : int
        Коефіцієнт зменшення (наприклад, 2 → fs/2).
    method : str
        'butter' — класичний фільтр Баттерворта,
        'poly'   — високоточний FIR через resample_poly.
    order : int
        Порядок фільтра (для butter).

    Повертає
    --------
    y : ndarray
        Децимований сигнал.
    fs_new : float
        Нова частота дискретизації.
    """
    if factor <= 1:
        return np.asarray(x), fs

    if method == "butter":
        fc = 0.45 * (fs / factor)  # 0.45 Nyquist нового fs
        sos = butter(order, fc / (fs / 2), btype="low", output="sos")
        xf = sosfiltfilt(sos, x)
        y = xf[::factor]
    elif method == "poly":
        # polyphase FIR ресемплінг — краща якість, але повільніше
        y = resample_poly(x, up=1, down=factor)
    else:
        raise ValueError("method must be 'butter' or 'poly'")

    return y, fs / factor
