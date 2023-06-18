import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Configuración de la detección de manos
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Variables para el tamaño deseado de la ventana
window_width = 1000
window_height = 800

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:

    hand_center_x = 0
    hand_center_y = 0
    hand_status = "Abierta"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB y procesarla con Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Dibujar los puntos de referencia de las manos en la imagen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener la posición del centro de la mano
                hand_center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                hand_center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

                # Mover el mouse con la mano
                pyautogui.moveTo(hand_center_x, hand_center_y)

                # Determinar el estado de la mano (abierta o cerrada)
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_IP].x and \
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x > hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_MCP].x:
                    hand_status = "Cerrada"
                else:
                    hand_status = "Abierta"

                # Hacer clic izquierdo al cerrar la mano
                if hand_status == "Cerrada":
                    pyautogui.click(button='left')
                    print("Se hizo clic izquierdo")

        # Agregar información a la ventana
        cv2.putText(frame, f"Estado de la mano: {hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Coordenadas de la mano: ({hand_center_x}, {hand_center_y})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Redimensionar el frame a las dimensiones deseadas
        frame = cv2.resize(frame, (window_width, window_height))

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()








