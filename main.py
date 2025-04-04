import cv2  # type: ignore
import face_recognition
from face_detection import FaceDetector
from face_registration import FaceRegistrar
from face_embeddings import get_face_embedding

def draw_overlay(frame, instructions,
                 panel_pos=(0, 0),
                 panel_size=(300, None),
                 instructions_offset=(10, 30),
                 line_height=30,
                 instructions_font_scale=0.7,
                 instructions_thickness=2,
                 instructions_color=(255, 255, 255),
                 panel_color=(50, 50, 50),
                 panel_alpha=0.6):
    """
    Dibuja un panel semitransparente en la posición y tamaño indicados,
    mostrando únicamente las instrucciones.
    """
    h_frame, w_frame = frame.shape[:2]
    x, y = panel_pos
    panel_width, panel_height = panel_size
    if panel_height is None:
        panel_height = h_frame

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), panel_color, -1)
    frame = cv2.addWeighted(overlay, panel_alpha, frame, 1 - panel_alpha, 0)

    inst_x, inst_y = instructions_offset
    for i, line in enumerate(instructions):
        cv2.putText(frame, line, (x + inst_x, y + inst_y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, instructions_font_scale, instructions_color, instructions_thickness)
    return frame

def recognize_faces(frame, face_detector, registered_faces, tolerance=0.6):
    """
    Para cada rostro detectado en 'frame', calcula el embedding y lo compara
    con los almacenados en 'registered_faces'. Devuelve una lista de tuplas (nombre, box).
    Si no hay coincidencia, el nombre será "Desconocido".
    """
    recognized = []
    boxes = face_detector.get_face_boxes(frame)
    for box in boxes:
        (startX, startY, endX, endY) = box
        face_roi = frame[startY:endY, startX:endX]
        embedding = get_face_embedding(face_roi)
        name = "Desconocido"
        if embedding is not None and registered_faces:
            for reg_name, reg_embedding in registered_faces.items():
                matches = face_recognition.compare_faces([reg_embedding], embedding, tolerance=tolerance)
                if matches[0]:
                    name = reg_name
                    break
        recognized.append((name, box))
    return recognized

def main():
    # Inicializa el detector de rostros
    face_detector = FaceDetector(
        prototxt_path="deploy.prototxt",
        caffemodel_path="res10_300x300_ssd_iter_140000.caffemodel",
        confidence_threshold=0.5
    )
    # Inicializa el registrador (que carga las caras registradas persistentes)
    face_registrar = FaceRegistrar(face_detector)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cv2.namedWindow("Deteccion de Rostros", cv2.WINDOW_NORMAL)

    # Configura el panel para las instrucciones en la parte inferior del frame
    ret, temp_frame = cap.read()
    if not ret:
        print("No se pudo leer un frame para configurar el overlay.")
        return
    frame_height, frame_width = temp_frame.shape[:2]
    panel_height = 100
    panel_position = (0, frame_height - panel_height)  # Panel en la parte inferior
    panel_size = (frame_width, panel_height)
    instructions_offset = (10, 30)
    line_height = 30

    instructions = [
        "Instrucciones:",
        "q: Salir",
        "r: Registrar"
    ]

    # Para mejorar el rendimiento, reconocemos rostros cada 'skip_frames'
    skip_frames = 1
    frame_counter = 0
    recognized_faces = []  # Almacena el resultado del reconocimiento

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir el frame. Saliendo...")
            break

        frame_counter += 1

        # Detección de rostros (rápida)
        frame_with_faces = face_detector.detect_faces(frame.copy())

        # Reconocimiento de rostros solo cada skip_frames para mejorar los fps
        if frame_counter % skip_frames == 0:
            recognized_faces = recognize_faces(frame, face_detector, face_registrar.registered_faces)

        # Etiqueta los rostros reconocidos
        for name, (startX, startY, endX, endY) in recognized_faces:
            cv2.putText(frame_with_faces, name, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Agrega el panel de instrucciones en la parte inferior (no se superpone a la imagen principal)
        frame_with_overlay = draw_overlay(frame_with_faces, instructions,
                                          panel_pos=panel_position,
                                          panel_size=panel_size,
                                          instructions_offset=instructions_offset,
                                          line_height=line_height)

        cv2.imshow("Deteccion de Rostros", frame_with_overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Registra la cara (pide el nombre por consola, calcula el embedding y lo almacena)
            message = face_registrar.register_face(frame)
            print(message)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
