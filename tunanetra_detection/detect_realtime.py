import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load model
model = load_model("models/sign_language_model.h5")
labels = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

# Konfigurasi
JEDA_THRESHOLD = 1.5  # Detik tanpa tangan dianggap spasi
HOLD_TIME = 1.5       # Minimal detik untuk konfirmasi huruf
MAX_DISPLAY_LENGTH = 10  # Jumlah karakter maksimum yang ditampilkan di layar

# Variabel state
current_text = []      # Menyimpan seluruh teks
current_word = []      # Menyimpan kata sedang diketik
current_prediction = None
prediction_start_time = None
last_detection_time = time.time()  # Inisialisasi variabel yang missing
scroll_offset = 0      # Untuk teks panjang

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

def get_display_text(full_text, offset, max_length):
    """Potong teks untuk ditampilkan dengan scroll"""
    end_idx = min(offset + max_length, len(full_text))
    return full_text[offset:end_idx]

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

        # Cek apakah huruf sudah bertahan cukup lama
        if prediction_start_time and (time.time() - prediction_start_time) >= HOLD_TIME:
            current_word.append(predicted_class)
            prediction_start_time = None  # Reset agar bisa deteksi huruf sama lagi
            last_detection_time = time.time()  # Update waktu terakhir deteksi
        
        # Tampilkan prediksi saat ini
        cv2.putText(frame, f"Prediksi: {predicted_class}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Progress bar hold time
        hold_progress = min(1.0, (time.time() - prediction_start_time) / HOLD_TIME if prediction_start_time else 0.0)
        cv2.rectangle(frame, (10, 80), (int(10 + 200 * hold_progress), 100), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 80), (210, 100), (255, 255, 255), 1)
        cv2.putText(frame, "Hold to confirm", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        current_prediction = None
        prediction_start_time = None
        
        # Cek jeda untuk menyelesaikan kata
        if time.time() - last_detection_time > JEDA_THRESHOLD and current_word:
            current_text.extend(current_word)
            current_text.append(" ")  # Tambah spasi
            current_word.clear()
            print("Teks saat ini:", "".join(current_text))

    # Gabungkan teks yang sedang diketik dengan yang sudah dikonfirmasi
    full_text = current_text + current_word
    display_text = get_display_text(full_text, scroll_offset, MAX_DISPLAY_LENGTH)
    
    # Tampilkan teks
    cv2.putText(frame, f"Sedang diketik: {''.join(current_word)}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Teks: {display_text}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Jika teks panjang, tampilkan indikator scroll
    if len(full_text) > MAX_DISPLAY_LENGTH:
        cv2.putText(frame, "[...]", (10 + 200, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Reset semua teks
        current_text.clear()
        current_word.clear()
        current_prediction = None
        prediction_start_time = None
        scroll_offset = 0
        last_detection_time = time.time()  # Reset juga waktu deteksi
        print("Teks di-reset!")
    elif key == ord('q'):  # Keluar
        break
    elif key == ord('a'):  # Scroll kiri
        scroll_offset = max(0, scroll_offset - 1)
    elif key == ord('d'):  # Scroll kanan
        scroll_offset = min(len(full_text) - MAX_DISPLAY_LENGTH, scroll_offset + 1)

    cv2.imshow("Deteksi Bahasa Isyarat", frame)

cap.release()
cv2.destroyAllWindows()