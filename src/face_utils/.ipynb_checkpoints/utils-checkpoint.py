# src/face_utils/utils.py

import os
from pathlib import Path

def get_model_paths():
    """
    Возвращает абсолютные пути к модели и конфигурации.
    Ищет папку 'models' в корне проекта (рядом с папкой 'src').
    Работает независимо от того, откуда запущен скрипт или ноутбук.
    """
    # __file__ = .../src/face_utils/utils.py
    # parent → face_utils
    # parent.parent → src
    # parent.parent.parent → корень проекта
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models"

    config_path = model_dir / "deploy.prototxt"
    model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path.absolute()}")
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path.absolute()}")

    return str(config_path), str(model_path)


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    import cv2
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)