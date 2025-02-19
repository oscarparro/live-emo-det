import cv2
import face_recognition

def get_face_embedding(face_image):
    """
    Convierte la imagen (en BGR) a RGB y calcula el embedding usando face_recognition.
    Devuelve un vector (128 dimensiones) o None si no se detecta ninguna cara.
    """
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None
