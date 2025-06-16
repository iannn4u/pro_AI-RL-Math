import cv2
import numpy as np
import joblib
import mediapipe as mp

# Muat model
model = joblib.load('model/model.pkl')
le = joblib.load('model/label_encoder.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Ekstrak landmark
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            # Prediksi
            pred = model.predict([landmarks])
            pred_label = le.inverse_transform(pred)[0]
            
            # Tampilkan hasil
            cv2.putText(frame, f"Prediksi: {pred_label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Pengenalan Huruf Isyarat', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()