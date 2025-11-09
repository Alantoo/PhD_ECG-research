"""
EnsembleSignalClassifier:
  Призначення:
    - Завантажує усі збережені класифікатори (pipeline .joblib) з директорії models_dir.
    - Проганяє переданий сигнал (один або кілька циклів) через кожну модель.
    - Агрегує всі предикти та визначає фінальний клас через мажоритарне голосування.
  Основні етапи:
    1. Виявлення доступних файлів моделей (*.joblib).
    2. Перетворення test_data у 2D numpy масив (авто-reshape для 1D).
    3. Передбачення кожною моделлю, збір усіх предиктів у один список.
    4. Обчислення кількості появ кожної мітки та вибір мітки з максимальною частотою.
    5. Повернення одного числа (тип сигналу) або None, якщо передбачити неможливо.
  Метод:
    classifySignal(test_data, model_paths=None):
      - test_data: масив ознак (1D або 2D).
      - model_paths: опціонально список шляхів до моделей; якщо None — автопошук.
      - Повертає: int (клас) або None.
"""
import os
import joblib
import numpy as np
from loguru import logger

class EnsembleSignalClassifier:
    """Ансамблевий класифікатор сигналу.
    Логіка:
      - _load_model_paths: визначає список моделей.
      - classifySignal:
          * нормалізує форму даних
          * збирає предикти всіх моделей
          * застосовує мажоритарне голосування.
    Повертає один інт — фінальний прогноз класу або None при відсутності даних/моделей."""
    def __init__(self, models_dir="trained_classifiers"):
        self.models_dir = models_dir

    def _load_model_paths(self, model_paths):
        if model_paths is not None and len(model_paths) > 0:
            return model_paths
        if not os.path.isdir(self.models_dir):
            logger.warning(f"Models dir '{self.models_dir}' does not exist.")
            return []
        paths = [
            os.path.join(self.models_dir, f)
            for f in os.listdir(self.models_dir)
            if f.endswith(".joblib")
        ]
        if not paths:
            logger.warning("No model files found.")
        return paths

    def classifySignal(self, test_data, model_paths=None):
        if test_data is None:
            logger.warning("test_data is None.")
            return None
        arr = np.asarray(test_data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            logger.warning("test_data has invalid shape.")
            return None

        paths = self._load_model_paths(model_paths)
        if not paths:
            return None

        all_predictions = []
        for path in paths:
            try:
                pipeline = joblib.load(path)
            except Exception as e:
                logger.error(f"Failed to load model {path}: {e}")
                continue
            try:
                preds = pipeline.predict(arr)
            except Exception as e:
                logger.error(f"Failed to predict with model {path}: {e}")
                continue
            all_predictions.extend(preds.tolist())
            logger.debug(f"{os.path.basename(path)} predicted distribution: {np.unique(preds, return_counts=True)}")

        if not all_predictions:
            logger.warning("No predictions obtained.")
            return None

        # Majority vote
        unique, counts = np.unique(all_predictions, return_counts=True)
        majority_label = int(unique[np.argmax(counts)])
        logger.debug(f"Majority vote result: label={majority_label}, counts={dict(zip(unique, counts))}")
        return majority_label
