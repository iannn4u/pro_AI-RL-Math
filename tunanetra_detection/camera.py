import cv2
import os

# Konfigurasi
huruf = 'A'  # Ubah ini sesuai huruf yang ingin dikumpulkan (A, B, C, D)
jumlah_gambar = 50  # Jumlah gambar per huruf
folder_dataset = 'dataset'

# Buat folder jika belum ada
os.makedirs(f'{folder_dataset}/{huruf}', exist_ok=True)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
counter = 0

print(f"Pengambilan data untuk huruf {huruf}. Tekan 's' untuk ambil gambar, 'q' untuk keluar")

while counter < jumlah_gambar:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Mirror frame
    frame = cv2.flip(frame, 1)
    
    # Tampilkan instruksi
    cv2.putText(frame, f"Kumpulkan: Huruf {huruf}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gambar: {counter}/{jumlah_gambar}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Tekan 's' untuk ambil gambar", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Pengambilan Dataset', frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Simpan gambar
        cv2.imwrite(f'{folder_dataset}/{huruf}/{counter}.jpg', frame)
        print(f"Gambar {counter} disimpan")
        counter += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Pengambilan data selesai!")