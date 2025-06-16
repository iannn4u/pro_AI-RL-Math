import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# Load dataset
data = np.load("dataset_landmarks.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Normalisasi
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Model CNN Sederhana
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas: A, B, C
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
model.save("sign_language_model.h5")

# Di akhir train_model.py
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi Test: {accuracy*100:.2f}%")