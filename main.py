import cv2
from face_detection import FaceDetector
from face_registration import FaceRegistrar

def main():
    # Inicializa el detector de rostros
    face_detector = FaceDetector(
        prototxt_path="deploy.prototxt",
        caffemodel_path="res10_300x300_ssd_iter_140000.caffemodel",
        confidence_threshold=0.5
    )
    # Inicializa el registrador, pasándole el detector
    face_registrar = FaceRegistrar(face_detector)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    print("Presiona 'q' para salir, 'r' para registrar cara...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir el frame, saliendo...")
            break

        # Se obtiene una copia del frame para dibujar detecciones
        frame_with_faces = face_detector.detect_faces(frame.copy())
        
        # Si se ha registrado una cara, se anota el nombre en la detección
        if face_registrar.registered_name is not None:
            boxes = face_detector.get_face_boxes(frame)
            if boxes:
                # Selecciona la cara más grande (suponemos que es la registrada)
                largest_box = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
                (startX, startY, endX, endY) = largest_box
                # Dibuja el nombre sobre el bounding box
                cv2.putText(frame_with_faces, face_registrar.registered_name, 
                            (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)

        cv2.imshow("Detección de Rostros", frame_with_faces)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Llama a la función de registro de cara
            face_registrar.register_face(frame)
            # Cierra la ventana de registro (si se abrió)
            cv2.destroyWindow("Cara para registrar")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
