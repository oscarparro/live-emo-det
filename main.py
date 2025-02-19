import cv2  # type: ignore
from face_detection import FaceDetector
from face_registration import FaceRegistrar

def draw_overlay(frame,
                 instructions,
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

    # Crea un overlay para el panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), panel_color, -1)
    frame = cv2.addWeighted(overlay, panel_alpha, frame, 1 - panel_alpha, 0)

    # Dibuja las instrucciones en el panel
    inst_x, inst_y = instructions_offset
    for i, line in enumerate(instructions):
        cv2.putText(
            frame,
            line,
            (x + inst_x, y + inst_y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            instructions_font_scale,
            instructions_color,
            instructions_thickness
        )

    return frame

def main():
    # Inicializa el detector y el registrador de rostros.
    face_detector = FaceDetector(
        prototxt_path="deploy.prototxt",
        caffemodel_path="res10_300x300_ssd_iter_140000.caffemodel",
        confidence_threshold=0.5
    )
    face_registrar = FaceRegistrar(face_detector)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cv2.namedWindow("Deteccion de Rostros", cv2.WINDOW_NORMAL)
    # Si quieres forzar un tamaño específico, descomenta y ajusta:
    # cv2.resizeWindow("Deteccion de Rostros", 1000, 900)

    # Parámetros para el panel de instrucciones
    panel_position = (0, 0)        # posición del panel (x, y)
    panel_size = (200, 100)        # tamaño del panel (ancho, alto)
    instructions_offset = (10, 30) # desplazamiento para instrucciones dentro del panel
    line_height = 30

    instructions = [
        "Instrucciones",
        "q: Salir",
        "r: Registrar"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir el frame. Saliendo...")
            break

        # Detecta rostros en una copia del frame
        frame_with_faces = face_detector.detect_faces(frame.copy())

        # Si ya se ha registrado una cara, muestra el nombre sobre el rostro más grande
        if face_registrar.registered_name is not None:
            boxes = face_detector.get_face_boxes(frame)
            if boxes:
                largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                (startX, startY, endX, endY) = largest_box
                cv2.putText(
                    frame_with_faces,
                    face_registrar.registered_name,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        # Agrega el overlay con instrucciones (ya sin logs)
        frame_with_overlay = draw_overlay(
            frame_with_faces,
            instructions,
            panel_pos=panel_position,
            panel_size=panel_size,
            instructions_offset=instructions_offset,
            line_height=line_height
        )

        cv2.imshow("Deteccion de Rostros", frame_with_overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Registra la cara (pide el nombre por consola)
            face_registrar.register_face(frame)
            cv2.destroyWindow("Cara para registrar")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
