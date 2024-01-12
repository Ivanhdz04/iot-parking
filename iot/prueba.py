import cv2
import numpy as np

def process_frame(frame, reference_frame, threshold=30):
    # Convierte los frames a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    # Suaviza las imágenes para reducir el ruido
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    gray_reference = cv2.GaussianBlur(gray_reference, (5, 5), 0)

    # Calcula la diferencia entre el frame actual y el frame de referencia
    diff = cv2.absdiff(gray_reference, gray_frame)

    # Aplica un umbral adaptativo para resaltar las diferencias
    _, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def main():
    # Video de entrada (reemplaza 'nombre_del_video.mp4' con el nombre de tu archivo)
    video_path = 'carPark.mp4'

    # Abre el video
    cap = cv2.VideoCapture(video_path)

    # Lee el primer frame para establecer el frame de referencia
    ret, reference_frame = cap.read()
    if not ret:
        print("Error al leer el video.")
        return

    while True:
        # Lee un nuevo frame
        ret, frame = cap.read()
        if not ret:
            break

        # Procesa el frame para detectar cambios
        processed_frame = process_frame(frame, reference_frame)

        # Encuentra contornos en el frame procesado
        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibuja recuadros verdes o rojos alrededor de los contornos (espacios vacíos u ocupados)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area < 500:  # Ajusta el umbral según tus necesidades
                # Dibuja en rojo si el área es menor al umbral
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                # Dibuja en verde si el área es mayor al umbral
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Muestra el resultado
        cv2.imshow('Detección de Espacios', frame)

        # Sale del bucle si se presiona 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Libera los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
