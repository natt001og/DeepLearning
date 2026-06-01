import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =========================
# 1. CARGA DE DATOS
# =========================
df = pd.read_csv("dataset.csv")

# Separar features y target
X = df.drop(columns=['Severity'])
y = df['Severity']

# (IMPORTANTE) Ajustar labels si empiezan en 1
# TensorFlow espera clases desde 0 hasta N-1
# Si tus clases son [1,2,3,4], las convertimos a [0,1,2,3]
y = y - 1

# =========================
# 2. SPLIT 98 / 1 / 1
# =========================

# Primero: train_full vs test (1%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42, stratify=y
)

# Luego: split train vs validation (1% del total ≈ 1.01% de train_full)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.0101,
    random_state=42,
    stratify=y_train_full
)

# Liberar memoria
del X_train_full, y_train_full

# =========================
# 3. MODELO DNN
# =========================

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu'),
    #layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    #layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),

    layers.Dense(num_classes, activation='softmax')
])

# =========================
# 4. COMPILACIÓN
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. ENTRENAMIENTO
# =========================

print("Entrenando...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=2048
)

# =========================
# 6. EVALUACIÓN
# =========================

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")


# 6. GRÁFICAS PARA DETECTAR OVERFITTING / UNDERFITTING
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.show()