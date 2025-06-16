import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7  # Tingkatkan jika deteksi terlalu sensitif
)

def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Gagal membaca gambar: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if not results.multi_hand_landmarks:
        print(f"Tangan tidak terdeteksi di: {image_path}")
        return None
    landmarks = results.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()


# Load dataset
X, y = [], []
dataset_dir = "dataset"
for label, class_name in enumerate(sorted(os.listdir(dataset_dir))):
    class_dir = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        landmarks = extract_hand_landmarks(image_path)
        if landmarks is not None:
            X.append(landmarks)
            y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.savez("dataset_landmarks.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Tambahkan ini sebelum train_test_split
print(f"Total sampel yang berhasil diproses: {len(X)}")
if len(X) < 2:
    raise ValueError("Dataset terlalu kecil (minimal 2 sampel). Periksa gambar atau ekstraksi landmark.")