# src/face_utils/detector.py

import cv2
import numpy as np
from .utils import get_model_paths  # ← новая функция


class FaceDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        config_path, model_path = get_model_paths()  # ← не скачивает, только проверяет
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    def detect(self, image):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Изображение должно быть RGB с 3 каналами.")

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 117.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes