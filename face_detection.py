import cv2 # type: ignore
import numpy as np

class FaceDetector:
    def __init__(self, prototxt_path, caffemodel_path, confidence_threshold=0.5):
        """
        Inicializa la red de detección de rostros.
        """
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        self.confidence_threshold = confidence_threshold

    def detect_faces(self, frame):
        """
        Dibuja un rectángulo en cada rostro detectado y devuelve el frame.
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return frame

    def get_face_boxes(self, frame):
        """
        Devuelve una lista de bounding boxes (startX, startY, endX, endY)
        de las caras detectadas en el frame.
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))
        return boxes
