from insightface.app import FaceAnalysis
import cv2
import numpy as np


class InsightFaceDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ожидалось RGB-изображение.")

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = self.app.get(img_bgr)

        boxes = []
        for face in faces:
            if face.det_score > self.confidence_threshold:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes