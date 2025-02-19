import cv2
from face_embeddings import get_face_embedding
from face_storage import load_registered_faces, save_registered_faces

class FaceRegistrar:
    def __init__(self, face_detector):
        """
        Inicializa el registrador de caras.
        Carga el diccionario de rostros registrados para conservar la información entre sesiones.
        """
        self.face_detector = face_detector
        self.registered_faces = load_registered_faces()  # {nombre: embedding}

    def register_face(self, frame):
        """
        Detecta la(s) cara(s) en el frame y, si se detecta exactamente una,
        solicita el nombre, calcula su embedding y lo almacena en el diccionario persistente.
        Devuelve un mensaje de resultado.
        """
        boxes = self.face_detector.get_face_boxes(frame)
        if not boxes:
            return "No se detectaron caras para registrar."
        if len(boxes) > 1:
            return "Demasiadas personas en el frame para registrar. Asegúrate de que solo haya un rostro."

        (startX, startY, endX, endY) = boxes[0]
        face_roi = frame[startY:endY, startX:endX]

        name = input("Ingresa tu nombre para registrar la cara: ")

        embedding = get_face_embedding(face_roi)
        if embedding is None:
            return "No se pudo calcular el embedding de la cara."

        self.registered_faces[name] = embedding
        save_registered_faces(self.registered_faces)
        return f"Cara registrada para: {name}"
