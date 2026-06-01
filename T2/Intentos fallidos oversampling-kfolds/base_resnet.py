import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. CARGA DE DATOS (ESTÁNDAR)
# ==========================================
base_path = "datos_particionados"  

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/train", image_size=(128, 128), batch_size=32, shuffle=True, crop_to_aspect_ratio=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/val", image_size=(128, 128), batch_size=32, shuffle=True, crop_to_aspect_ratio=True
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/test", image_size=(128, 128), batch_size=32, shuffle=False, crop_to_aspect_ratio=True
)

class_names = train_ds.class_names
num_clases = len(class_names)  

# ==========================================
# 2. ARQUITECTURA RESNET ESTÁNDAR (Feature Extraction Puro)
# ==========================================
preprocess_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input)

resnet_base = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Congelamos completamente la base preentrenada
resnet_base.trainable = False 

# Construimos el modelo sin regularizadores artificiales ni alteraciones de datos
model_resnet = Sequential([
    Input(shape=(128, 128, 3)),
    preprocess_layer,        
    resnet_base,             
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'), 
    Dense(num_clases, activation='softmax') 
])

model_resnet.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_resnet.summary()

# ==========================================
# 3. ENTRENAMIENTO ESTÁNDAR FIJO (A 10 Épocas)
# ==========================================
print("\nIniciando Entrenamiento Fijo y Estándar...")
start_time = time.time()

history = model_resnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10 # Correrá las 10 épocas sin detenerse y con pesos de error uniformes
)

end_time = time.time()
print(f"Tiempo de entrenamiento: {(end_time - start_time)/60:.2f} minutes")

# ==========================================
# 4. EVALUACIÓN Y GUARDADO DE GRÁFICOS
# ==========================================
output_dir = "resultados_resnet_estandar"
os.makedirs(output_dir, exist_ok=True)

test_loss, test_acc = model_resnet.evaluate(test_ds)
print(f"\n[RESULTADOS] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

# Guardar Modelo e Historial
model_resnet.save(f"{output_dir}/resnet_estandar.keras")
with open(f"{output_dir}/history_resnet_estandar.pkl", "wb") as f:
    pickle.dump(history.history, f)

# --- Gráfico Accuracy ---
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ResNet Estandar - Accuracy vs Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/grafico_accuracy_resnet_estandar.png", dpi=300)
plt.close()

# --- Gráfico Loss ---
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ResNet Estandar - Loss vs Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/grafico_loss_resnet_estandar.png", dpi=300)
plt.close()

# --- Matriz de Confusión ---
y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

preds = model_resnet.predict(test_ds)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - ResNet Estandar")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{output_dir}/matriz_confusion_resnet_estandar.png", dpi=300)
plt.close()

# --- Reporte ---
print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))