import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
base_path = "datos_particionados"  

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/train",
    image_size=(128, 128),
    batch_size=32,
    crop_to_aspect_ratio=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/val",
    image_size=(128, 128),
    batch_size=32,
    crop_to_aspect_ratio=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/test",
    image_size=(128, 128),
    batch_size=32,
    crop_to_aspect_ratio=True,
    shuffle=False  
)

class_names = train_ds.class_names
num_clases = len(class_names)

output_dir = "resultados_googlenet_dos_fases"
os.makedirs(output_dir, exist_ok=True)

# Capa de preprocesamiento específica para InceptionV3
preprocess_layer = tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input)

# Carga de InceptionV3 (GoogLeNet) sin el clasificador original
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# ==========================================
# FASE 1: FEATURE EXTRACTION (Base 100% Congelada)
# ==========================================
# MODIFICACIÓN CLAVE: Aquí congelamos COMPLETAMENTE el modelo base igual que en ResNet
for layer in base_model.layers:
    layer.trainable = False

# Construimos la estructura secuencial
model_inception = Sequential([
    preprocess_layer,
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_clases, activation='softmax')
])

# Compilamos la Fase 1 con el optimizador Adam y Learning Rate estándar (0.001)
model_inception.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando Fase 1: Feature Extraction (Clasificador superior)...")
start_time_1 = time.time()
history_fase1 = model_inception.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # 10 épocas para estabilizar las capas nuevas
)
end_time_1 = time.time()
print(f"Tiempo entrenamiento Fase 1: {(end_time_1 - start_time_1)/60:.2f} minutos")


# ==========================================
# FASE 2: FINE-TUNING (Descongelar últimas 30 capas)
# ==========================================
# MODIFICACIÓN CLAVE: Descongelamos QUIRÚRGICAMENTE solo las últimas 30 capas
for layer in base_model.layers[-30:]:
    layer.trainable = True

# MODIFICACIÓN CLAVE: Recompilamos con el Learning Rate ULTRA BAJO (1e-5) para proteger el conocimiento
model_inception.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nCapas totales en InceptionV3: {len(base_model.layers)}")
print(f"Capas que se acaban de abrir para Fine-Tuning: {sum(layer.trainable for layer in base_model.layers)}")

print("\nIniciando Fase 2: Fine-tuning...")
start_time_2 = time.time()
history_fase2 = model_inception.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15  # 15 épocas extras de ajuste fino fino, igual que en tu ResNet
)
end_time_2 = time.time()
print(f"Tiempo entrenamiento Fase 2: {(end_time_2 - start_time_2)/60:.2f} minutos")


# ==========================================
# EVALUACIÓN Y GUARDADO DE RESULTADOS
# ==========================================
print("\nEvaluando GoogLeNet/Inception en el dataset de Test...")
test_loss, test_accuracy = model_inception.evaluate(test_ds)
print(f"\n[RESULTADOS FINALES] Test Accuracy: {test_accuracy:.4f} | Test Loss: {test_loss:.4f}")

tiempo_total = (end_time_2 - start_time_1) / 60
print(f"\nTiempo TOTAL de todo el proceso: {tiempo_total:.2f} minutos")

# Guardar el modelo entrenado
model_inception.save(f"{output_dir}/googlenet_dos_fases.keras")

# MODIFICACIÓN CLAVE: Concatenamos los historiales de ambas fases de forma limpia
history_total = {
    "accuracy": history_fase1.history['accuracy'] + history_fase2.history['accuracy'],
    "val_accuracy": history_fase1.history['val_accuracy'] + history_fase2.history['val_accuracy'],
    "loss": history_fase1.history['loss'] + history_fase2.history['loss'],
    "val_loss": history_fase1.history['val_loss'] + history_fase2.history['val_loss']
}

with open(f"{output_dir}/history_googlenet_dos_fases.pkl", "wb") as f:
    pickle.dump(history_total, f)

# Generar Gráfico de Accuracy con la línea roja divisoria
plt.figure(figsize=(6, 4))
plt.plot(history_total['accuracy'], label='Train Accuracy')
plt.plot(history_total['val_accuracy'], label='Val Accuracy')
plt.axvline(x=len(history_fase1.history['accuracy'])-1, color='r', linestyle='--', label='Inicio Fine-tuning')
plt.title('GoogLeNet/Inception - Accuracy vs Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/grafico_accuracy_googlenet2.png', dpi=300)
plt.close()

# Generar Gráfico de Loss con la línea roja divisoria
plt.figure(figsize=(6, 4))
plt.plot(history_total['loss'], label='Train Loss')
plt.plot(history_total['val_loss'], label='Val Loss')
plt.axvline(x=len(history_fase1.history['loss'])-1, color='r', linestyle='--', label='Inicio Fine-tuning')
plt.title('GoogLeNet/Inception - Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/grafico_loss_googlenet2.png', dpi=300)
plt.close()

# Generar Matriz de Confusión limpia
print("\nGenerando matriz de confusión...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
preds = model_inception.predict(test_ds)
y_pred = np.argmax(preds, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - GoogLeNet/Inception Fases")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/matriz_confusion_googlenet2.png', dpi=300)
plt.close()

# Reporte final por consola
print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
print(f"\n✔ ¡PROCESO COMPLETADO! Todo guardado de forma equivalente en: {os.path.abspath(output_dir)}")