from scipy.signal import find_peaks, savgol_filter
import numpy as np
from scipy.interpolate import interp1d

class Simulation:
    def __init__(self):
        pass


    def gen_cycle(self, rhythm_data, variance_data, mean_data, count):
        last_time = 0
        points = list()
        for iter in range(count):
            # Інтерполяція для приведення до спільного часу
            t = np.linspace(0, 1, 500)  # Спільний часовий інтервал

            rhythm_interp = interp1d(rhythm_data[0], rhythm_data[1], kind='cubic', fill_value="extrapolate")
            mean_interp = interp1d(mean_data[0], mean_data[1], kind='cubic', fill_value="extrapolate")
            variance_interp = interp1d(variance_data[0], variance_data[1], kind='cubic', fill_value="extrapolate")

            # Отримуємо значення на всьому часовому інтервалі
            rhythm = rhythm_interp(t)
            mean = mean_interp(t)
            variance = variance_interp(t)

            # Генерація циклічного сигналу
            ecg_signal = mean + np.sin(2 * np.pi * rhythm * t) + np.random.normal(0, np.sqrt(variance), len(t))

            postprocessed = savgol_filter(ecg_signal, window_length=13, polyorder=3)
            last_tval = 0
            for i in range(len(postprocessed)):
                time = t[i] + last_time
                value = postprocessed[i]
                last_tval = time
                points.append([time, value])

            last_time = last_tval

        return points