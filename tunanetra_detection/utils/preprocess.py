import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Gambar rusak/tidak terbaca: {image_path}")
        os.remove(image_path)  # Hapus gambar yang rusak
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    if not results.multi_hand_landmarks:
        print(f"❌ Tangan tidak terdeteksi di: {image_path}")
        os.remove(image_path)  # Hapus gambar tanpa tangan
        return None
    
    landmarks = results.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

def process_dataset(dataset_dir):
    X, y = [], []
    deleted_files = 0

    for label, class_name in enumerate(sorted(os.listdir(dataset_dir))):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"\n🔍 Memproses kelas: {class_name} (Label: {label})")
        class_images = os.listdir(class_dir)
        total_in_class = len(class_images)
        processed = 0

        for image_name in class_images:
            image_path = os.path.join(class_dir, image_name)
            landmarks = extract_hand_landmarks(image_path)
            
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
                processed += 1
            else:
                deleted_files += 1

        print(f"✅ Berhasil: {processed}/{total_in_class} gambar")

    return X, y, deleted_files

# Main execution
if __name__ == "__main__":
    dataset_dir = "dataset"
    
    print("🔄 Memulai ekstraksi landmark...")
    X, y, deleted_files = process_dataset(dataset_dir)
    
    if len(X) < 2:
        raise ValueError("\n❌ Dataset terlalu kecil setelah pembersihan. Periksa kualitas gambar!")
    
    print(f"\n📊 Hasil akhir:")
    print(f"- Total sampel valid: {len(X)}")
    print(f"- Total gambar dihapus: {deleted_files}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    np.savez("models/dataset_landmarks.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("\n💾 Data berhasil disimpan di models/dataset_landmarks.npz")