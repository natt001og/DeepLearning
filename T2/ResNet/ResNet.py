import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report


# Carga de datos
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

# Extrae los nombres de las clases para la matriz de confusión
class_names = train_ds.class_names
num_clases = len(class_names)  


# Definimos la función de preprocesamiento específica para ResNet50
preprocess_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input)

resNet_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Congelamos la base para el entrenamiento inicial (Feature Extraction)
for layer in resNet_model.layers:
    layer.trainable = False

# Construimos el modelo 
model_resnet = Sequential([
    preprocess_layer,  
    resNet_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_clases, activation='softmax') 
])

# Aquí inicia la primera etapa de entrenamiento
model_resnet.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Iniciando Fase 1: Feature Extraction...")
start_time_1 = time.time()
history_resnet = model_resnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
end_time_1 = time.time()
print(f"Tiempo de entrenamiento Fase 1: {end_time_1 - start_time_1:.2f} segundos")


# Inicio de la 2da etapa de entrenamiento: Fine-tuning
# Descongelar las últimas 30 capas
for layer in resNet_model.layers[-30:]:
    layer.trainable = True

model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando Fase 2: Fine-tuning...")
start_time_2 = time.time()
history_finetune = model_resnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)
end_time_2 = time.time()
print(f"Tiempo de entrenamiento Fase 2: {end_time_2 - start_time_2:.2f} segundos")


# Evaluación final en el dataset de Test
print("\nEvaluando en el dataset de Test...")
test_loss, test_acc = model_resnet.evaluate(test_ds)
print(f"\n[RESULTADOS FINALES] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

tiempo_total = (end_time_2 - start_time_1) / 60
print(f"\nTiempo TOTAL entrenamiento: {tiempo_total:.2f} minutos")


output_dir = "resultados_resnet"
os.makedirs(output_dir, exist_ok=True)

print("Guardando resultados en:", os.path.abspath(output_dir))

model_resnet.save(f"{output_dir}/resnet_model.keras")

# graficos de entrenamiento

history_total = {
    "accuracy": history_resnet.history['accuracy'] + history_finetune.history['accuracy'],
    "val_accuracy": history_resnet.history['val_accuracy'] + history_finetune.history['val_accuracy'],
    "loss": history_resnet.history['loss'] + history_finetune.history['loss'],
    "val_loss": history_resnet.history['val_loss'] + history_finetune.history['val_loss']
}

with open(f"{output_dir}/history_resnet.pkl", "wb") as f:
    pickle.dump(history_total, f)


#Accuracy
plt.figure(figsize=(6,4))
plt.plot(history_total['accuracy'], label='Train Accuracy')
plt.plot(history_total['val_accuracy'], label='Val Accuracy')
plt.axvline(x=len(history_resnet.history['accuracy'])-1, color='r', linestyle='--', label='Inicio Fine-tuning')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ResNet - Accuracy vs Epochs')
plt.legend()
plt.tight_layout()

plt.savefig(f"{output_dir}/grafico_accuracy_resnet_sdropout3.png", dpi=300)
plt.close()

print("✔ Guardado: grafico_accuracy_resnet_sdropout3.png")

#Loss
plt.figure(figsize=(6,4))
plt.plot(history_total['loss'], label='Train Loss')
plt.plot(history_total['val_loss'], label='Val Loss')
plt.axvline(x=len(history_resnet.history['loss'])-1, color='r', linestyle='--', label='Inicio Fine-tuning')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ResNet - Loss vs Epochs')
plt.legend()
plt.tight_layout()

plt.savefig(f"{output_dir}/grafico_loss_resnet_sdropout3.png", dpi=300)
plt.close()

print("✔ Guardado: grafico_loss_resnet_sdropout3.png")

# Matriz de Confusión
print("\nGenerando Matriz de Confusión...")

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

preds = model_resnet.predict(test_ds)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12,10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - ResNet")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(f"{output_dir}/matriz_confusion_resnet_sdropout3.png", dpi=300)
plt.close()

print("✔ Guardado: matriz_confusion_resnet_sdropout3.png")

#reporte de clasificación final
print("\nReporte de Clasificación:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

print("\n TODO GUARDADO CORRECTAMENTE EN:", os.path.abspath(output_dir))