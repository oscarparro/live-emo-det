import cv2
from face_detection import FaceDetector

def main():
    # Inicializa el detector de rostros
    face_detector = FaceDetector(
        prototxt_path="deploy.prototxt",
        caffemodel_path="res10_300x300_ssd_iter_140000.caffemodel",
        confidence_threshold=0.5
    )

    # Inicia la captura de video (cámara web)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    print("Presiona 'q' para salir...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir el frame, saliendo...")
            break

        # Detecta rostros en el frame
        frame_with_faces = face_detector.detect_faces(frame)

        # Muestra el resultado en pantalla
        cv2.imshow("Deteccion de Rostros", frame_with_faces)

        # Si se presiona 'q', se sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la cámara y cierra ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
