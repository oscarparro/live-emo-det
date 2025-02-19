import pickle
import os

DATABASE_FILE = "registered_faces.pkl"

def load_registered_faces():
    """
    Carga el diccionario de rostros registrados desde el archivo.
    Si el archivo no existe, devuelve un diccionario vacío.
    """
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return {}

def save_registered_faces(registered_faces):
    """
    Guarda el diccionario de rostros registrados en el archivo.
    """
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(registered_faces, f)
