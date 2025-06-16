import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model
model = load_model("sign_language_model.h5")
labels = {0: "A", 1: "B", 2: "C"}

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

    if results.multi_hand_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
        pred = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_class = labels[np.argmax(pred)]

        # Gambar hasil prediksi
        cv2.putText(frame, f"Prediksi: {predicted_class}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Deteksi Huruf Tunarungu", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()