import cv2

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
        Detecta las caras en el frame, selecciona la de mayor tamaño,
        muestra la región extraída y pide al usuario que ingrese su nombre.
        Guarda el nombre y la imagen de la cara registrada.
        """
        boxes = self.face_detector.get_face_boxes(frame)
        if not boxes:
            print("No se detectaron caras para registrar.")
            return

        # Selecciona la cara más grande (mayor área)
        largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        (startX, startY, endX, endY) = largest_box
        face_roi = frame[startY:endY, startX:endX]

        # Muestra la región de la cara en una ventana separada
        cv2.imshow("Cara para registrar", face_roi)
        cv2.waitKey(1)  # Solo para actualizar la ventana

        # Solicita el nombre al usuario (en consola)
        name = input("Ingresa tu nombre para registrar la cara: ")

        self.registered_name = name
        self.registered_face_image = face_roi
        print(f"Cara registrada para: {name}")
