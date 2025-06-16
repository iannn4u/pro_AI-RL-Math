import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Deteksi 2 tangan
mp_drawing = mp.solutions.drawing_utils

def mirror_hand_tracking():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kamera tidak dapat diakses")
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Konversi BGR ke RGB (MediaPipe membutuhkan RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses deteksi tangan
        results = hands.process(rgb_frame)
        
        # Gambar landmark tangan jika terdeteksi
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Warna hijau untuk landmarks
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Warna merah untuk koneksi
                )
        
        # Tampilkan jumlah jari terdeteksi (opsional)
        cv2.putText(frame, f"Tangan Terdeteksi: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Deteksi Jari - Mirror Mode', frame)
        
        # Keluar dengan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mirror_hand_tracking()