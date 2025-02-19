import cv2
import numpy as np

class FaceDetector:
    def __init__(self, prototxt_path, caffemodel_path, confidence_threshold=0.5):
        """
        Inicializa la red de detección de rostros y el umbral de confianza.
        """
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        self.confidence_threshold = confidence_threshold

    def detect_faces(self, frame):
        """
        Recibe un frame (imagen) y dibuja rectángulos alrededor de los rostros detectados.
        Devuelve el frame con los rectángulos dibujados.
        """
        (h, w) = frame.shape[:2]
        # Preprocesamiento de la imagen (blob)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            scalefactor=1.0, 
            size=(300, 300), 
            mean=(104.0, 177.0, 123.0)
        )

        # Pasar el blob por la red
        self.net.setInput(blob)
        detections = self.net.forward()

        # Recorrer todas las detecciones
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                # Escalar las coordenadas de la caja al tamaño original
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Dibujar el rectángulo alrededor del rostro
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return frame
