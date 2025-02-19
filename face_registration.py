import cv2 # type: ignore

class FaceRegistrar:
    def __init__(self, face_detector):
        """
        Inicializa el registrador de caras.
        Recibe una instancia de FaceDetector para reutilizar la detección.
        """
        self.face_detector = face_detector
        self.registered_name = None
        self.registered_face_image = None

    def register_face(self, frame):
        """
        Detecta las caras en el frame y, si se detecta exactamente una,
        muestra la imagen de la cara en una ventana, solicita el nombre y la registra.
        Si hay 0 o más de una cara, devuelve un mensaje de error.
        """
        boxes = self.face_detector.get_face_boxes(frame)
        if not boxes:
            return "No se detectaron caras para registrar."
        if len(boxes) > 1:
            return "Demasiadas personas en el frame para registrar. Asegúrate de que solo haya un rostro."

        (startX, startY, endX, endY) = boxes[0]
        face_roi = frame[startY:endY, startX:endX]

        cv2.imshow("Cara para registrar", face_roi)
        cv2.waitKey(1)  # Actualiza la ventana

        name = input("Ingresa tu nombre para registrar la cara: ")

        self.registered_name = name
        self.registered_face_image = face_roi
        return f"Cara registrada para: {name}"
