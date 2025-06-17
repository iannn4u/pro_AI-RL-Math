import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time

# Load model
model = load_model("models/sign_language_model.h5")
labels = {0: "A", 1: "B", 2: "C"}

# Konfigurasi
SEQUENCE_LENGTH = 5
JEDA_THRESHOLD = 1.5
HOLD_TIME = 1.5  # Minimal 2.5 detik untuk konfirmasi huruf
history = deque(maxlen=SEQUENCE_LENGTH)
last_detection_time = time.time()
current_word = []
current_prediction = None
prediction_start_time = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Deteksi huruf
    if results.multi_hand_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
        pred = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
        predicted_class = labels[np.argmax(pred)]

        # Jika prediksi berubah, reset timer
        if current_prediction != predicted_class:
            current_prediction = predicted_class
            prediction_start_time = time.time()

        # Cek apakah huruf sudah bertahan selama HOLD_TIME
        if prediction_start_time and (time.time() - prediction_start_time) >= HOLD_TIME:
            if not history or history[-1] != predicted_class:
                history.append(predicted_class)
                last_detection_time = time.time()
        
        cv2.putText(frame, f"Prediksi: {predicted_class}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Tampilkan progress hold time
        hold_progress = min(1.0, (time.time() - prediction_start_time) / HOLD_TIME if prediction_start_time else 0.0)
        cv2.rectangle(frame, (10, 80), (int(10 + 200 * hold_progress), 100), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 80), (210, 100), (255, 255, 255), 1)
        cv2.putText(frame, "Hold to confirm", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        current_prediction = None
        prediction_start_time = None
        # Cek jeda jika tidak ada tangan
        if time.time() - last_detection_time > JEDA_THRESHOLD and history:
            current_word.append("".join(history))
            history.clear()
            print("Kata sementara:", " ".join(current_word))

    # Tampilkan history dan kata
    cv2.putText(frame, f"History: {list(history)}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Kata: {' '.join(current_word)}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Reset kata dengan tombol 'r'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        current_word.clear()
        history.clear()
        current_prediction = None
        prediction_start_time = None
        print("Kata di-reset!")
    if key == ord('q'):
        break

    cv2.imshow("Deteksi Kata Bahasa Isyarat", frame)

cap.release()
cv2.destroyAllWindows()