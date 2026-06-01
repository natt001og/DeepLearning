import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report

#Carga de datos
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


# Definicion del modelo InceptionV3
#
preprocess_layer = tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input)

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Fine-tuning 
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Construcción de la arquitectura 
model_inception = Sequential([
    preprocess_layer,  # Escala las imágenes entre -1 y 1 automáticamente
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_clases, activation='softmax')  
])

# Mostrar resumen en consola
model_inception.summary()


# compilación del modelo y entrenamiento 

model_inception.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando entrenamiento de InceptionV3 (Fine-Tuning de 50 capas)...")
history_inception = model_inception.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# evaluación final en el dataset de Test

print("\nEvaluando InceptionV3 en el dataset de Test...")
test_loss_inception, test_accuracy_inception = model_inception.evaluate(test_ds)
print(f"\n[RESULTADOS FINALES INCEPTION] Test Accuracy: {test_accuracy_inception:.4f} | Test Loss: {test_loss_inception:.4f}")


# gráficos 

# Gráfico de Exactitud (Accuracy)
plt.figure(figsize=(6, 4))
plt.plot(history_inception.history['accuracy'], label='Train')
plt.plot(history_inception.history['val_accuracy'], label='Validation')
plt.title('Accuracy InceptionV3 Fine Tuning')
plt.ylabel('Accuracy')
plt.xlabel('Épocas')
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico de Pérdida (Loss)
plt.figure(figsize=(6, 4))
plt.plot(history_inception.history['loss'], label='Train')
plt.plot(history_inception.history['val_loss'], label='Validation')
plt.title('Loss InceptionV3 Fine Tuning')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.legend()
plt.tight_layout()
plt.show()


# matriz de confusión y reporte de clasificación por clase

print("\nGenerando análisis de errores por clase para InceptionV3...")

# Extraer etiquetas verdaderas de forma ordenada
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Obtener predicciones del modelo
preds = model_inception.predict(test_ds)
y_pred = np.argmax(preds, axis=1)

# Calcular matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Graficar la matriz de confusión
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,       
    fmt="d",          
    cmap="Purples",   
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - InceptionV3")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nReporte de Clasificación por Clase (InceptionV3):")
print(classification_report(y_true, y_pred, target_names=class_names))