"""
SavedModelsTester:
  Призначення:
    - Завантажує збережені класифікатори (pipeline з препроцесором та моделлю) з директорії models_dir.
    - Виконує предикт на переданих тестових даних data_test.
    - Порівнює предикти з true_labels та обчислює точність (accuracy_percent) для кожної моделі.
    - Повертає список словників з детальною інформацією (шлях до моделі, істинні мітки, предикти, точність).
  Основні кроки:
    1. Валідація test даних та міток.
    2. Автовизначення списку моделей, якщо model_paths не передано.
    3. Ітерація по моделях: завантаження + передбачення + розрахунок точності.
    4. Агрегація результатів у єдиний список.
  Метод:
    testData(model_paths, data_test, true_labels):
      - model_paths: список шляхів до моделей (.joblib) або None для автоматичного пошуку.
      - data_test: 2D масив ознак.
      - true_labels: 1D масив істинних класів.
      - Повертає: list[dict].
"""
import os
import joblib
import numpy as np
from loguru import logger

class SavedModelsTester:
    """Клас для тестування збережених моделей та обчислення точності на тестових даних."""
    def __init__(self, models_dir="trained_classifiers"):
        self.models_dir = models_dir

    def testData(self, model_paths, data_test, true_labels):
        if data_test is None or true_labels is None:
            logger.warning("Test data or true labels are None.")
            return []
        if not isinstance(data_test, np.ndarray):
            data_test = np.asarray(data_test)
        if not isinstance(true_labels, np.ndarray):
            true_labels = np.asarray(true_labels)
        if model_paths is None or len(model_paths) == 0:
            if not os.path.isdir(self.models_dir):
                logger.warning(f"Models dir '{self.models_dir}' does not exist.")
                return []
            model_paths = [
                os.path.join(self.models_dir, f)
                for f in os.listdir(self.models_dir)
                if f.endswith(".joblib")
            ]
        results = []
        for path in model_paths:
            try:
                pipeline = joblib.load(path)
            except Exception as e:
                logger.error(f"Failed to load model {path}: {e}")
                continue
            predicted = pipeline.predict(data_test)
            accuracy_percent = float(np.mean(predicted == true_labels) * 100.0)
            results.append({
                "model_path": path,
                "true_labels": true_labels,
                "predicted_labels": predicted,
                "accuracy_percent": accuracy_percent
            })
            logger.debug(f"Tested {os.path.basename(path)}: {len(predicted)} predictions, accuracy={accuracy_percent:.2f}%.")
        return results