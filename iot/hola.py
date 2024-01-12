import cv2
import numpy as np

def process_frame(frame, bg_subtractor, reference_frame, threshold=30):
    # Aplica la resta de fondo al frame actual
    fg_mask = bg_subtractor.apply(frame)

    # Aplica un umbral adaptativo para resaltar las diferencias
    _, thresholded = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)

    return thresholded

def main():
    # Video de entrada (reemplaza 'nombre_del_video.mp4' con el nombre de tu archivo)
    video_path = 'carPark.mp4'

    # Abre el video
    cap = cv2.VideoCapture(video_path)

    # Crea un objeto Background Subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Lee el primer frame para establecer el frame de referencia
    ret, reference_frame = cap.read()
    if not ret:
        print("Error al leer el video.")
        return

    # Tamaño de referencia para el cajón del estacionamiento (ajusta según tus necesidades)
    reference_size = 5000

    while True:
        # Lee un nuevo frame
        ret, frame = cap.read()
        if not ret:
            break

        # Procesa el frame para detectar cambios
        processed_frame = process_frame(frame, bg_subtractor, reference_frame)

        # Encuentra contornos en el frame procesado
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contadores de espacios vacíos y ocupados
        vacant_spaces = 0
        occupied_spaces = 0

        # Dibuja recuadros verdes o rojos alrededor de los contornos (espacios vacíos u ocupados)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Ajusta los valores según tus necesidades
            aspect_ratio = w / float(h)
            aspect_ratio_threshold = 2.0  # Puedes ajustar este umbral

            # Filtra contornos en función de su área y relación de aspecto
            if 500 < area < 5000 and 0.5 < aspect_ratio < aspect_ratio_threshold:
                # Dibuja en verde si el área está en el rango y tiene la relación ancho/alto adecuada
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                vacant_spaces += 1
            else:
                # Dibuja en rojo si no cumple con los criterios de área y relación de aspecto
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                occupied_spaces += 1

        # Muestra el resultado y la cantidad de espacios disponibles y ocupados
        cv2.putText(frame, f'Espacios Disponibles: {vacant_spaces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Detección de Espacios', frame)

        # Sale del bucle si se presiona 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Libera los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
