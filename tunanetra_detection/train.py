import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import mediapipe as mp

# Fungsi untuk ekstrak landmark
def extract_landmarks(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return None

# Muat dataset
def load_dataset(dataset_path):
    X, y = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y)

# Main
if __name__ == "__main__":
    # Muat data
    X, y = load_dataset('dataset')
    
    # Encode label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Latih model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluasi
    accuracy = model.score(X_test, y_test)
    print(f"Akurasi: {accuracy:.2f}")
    
    # Simpan model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')
    print("Model disimpan di folder 'model'")