import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Tidak dapat menerima frame")
        continue
    
    # Konversi BGR ke RGB dan flip horizontal (mirror)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Optimasi performa
    image.flags.writeable = False  # Matikan penulisan untuk mempercepat proses
    
    # Proses deteksi tangan
    results = hands.process(image)
    
    # Aktifkan kembali penulisan untuk menggambar
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Gambar landmark tangan jika terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark dan koneksi tangan
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hitung jumlah jari yang terangkat
            finger_count = 0
            fingers_up = []
            
            # Ambil landmark ujung jari dan sendi
            tip_ids = [4, 8, 12, 16, 20]  # Ibu jari, telunjuk, tengah, manis, kelingking
            pip_ids = [3, 6, 10, 14, 18]  # Sendi pangkal jari
            
            # Deteksi ibu jari (perhitungan khusus)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_pip = hand_landmarks.landmark[3]
            if tip_ids[0] == 4:  # Untuk tangan kanan
                if thumb_tip.x < thumb_pip.x:
                    finger_count += 1
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:  # Untuk tangan kiri
                if thumb_tip.x > thumb_pip.x:
                    finger_count += 1
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            # Deteksi jari lainnya
            for i in range(1, 5):
                tip = hand_landmarks.landmark[tip_ids[i]]
                pip = hand_landmarks.landmark[pip_ids[i]]
                
                if tip.y < pip.y:  # Jika ujung jari di atas sendi
                    finger_count += 1
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            # Tampilkan jumlah jari
            cv2.putText(image, f'Jari: {finger_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Deteksi Peace Sign (jari telunjuk dan tengah terangkat, lainnya menekuk)
            if (finger_count == 2 and 
                fingers_up[1] == 1 and  # Telunjuk
                fingers_up[2] == 1 and  # Tengah
                fingers_up[3] == 0 and  # Manis
                fingers_up[4] == 0):    # Kelingking
                cv2.putText(image, "Peace Sign!", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Deteksi Jari Tangan', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()