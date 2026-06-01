import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping  # <-- Importamos el Callback
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight


base_path = "datos_particionados"  

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/train",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True,          
    crop_to_aspect_ratio=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/val",
    image_size=(128, 128),      
    batch_size=32,
    shuffle=True,
    crop_to_aspect_ratio=True
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/test",
    image_size=(128, 128),
    batch_size=32,
    shuffle=False,
    crop_to_aspect_ratio=True
)

class_names = train_ds.class_names
num_clases = len(class_names)  
print(f"Dataset listo. Detectadas {num_clases} clases de peces.")

# cálculo de pesos de clase con suavizado para evitar sobre-corrección
print("\nExtrayendo etiquetas de entrenamiento para calcular pesos...")
y_train = []
for _, labels in train_ds:
    y_train.extend(labels.numpy())
y_train = np.array(y_train)

indices_clases, conteos = np.unique(y_train, return_counts=True)

total_muestras = len(y_train)
pesos_suavizados = total_muestras / np.sqrt(conteos)

pesos_suavizados = pesos_suavizados / np.mean(pesos_suavizados)
class_weights = dict(zip(indices_clases, pesos_suavizados))

print("\nPesos suavizados asignados (Evita sobre-corrección):")
for k, v in class_weights.items():
    print(f"Clase {k} ({class_names[k]}): {v:.2f}")


output_dir = "resultados_cnn"
os.makedirs(output_dir, exist_ok=True)

# Arquitectura CNN simple desde cero
model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  

    Dense(num_clases, activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Detiene el entrenamiento si el val_loss no mejora durante 3 épocas consecutivas
# y restaura los mejores pesos automáticamente.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)



#Entrenamiento
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,                     # Subido a 15, ya que early_stopping controlará el freno
    class_weight=class_weights,
    callbacks=[early_stopping]  
)

end_time = time.time()
training_time = end_time - start_time
print(f"\nTiempo entrenamiento: {training_time/60:.2f} minutos")

# Evaluación test
test_loss, test_acc = model.evaluate(test_ds)

print(f"\n[RESULTADOS CNN]")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")


model.save(f"{output_dir}/cnn_model_balanced.keras")
print("✔ Modelo guardado")

with open(f"{output_dir}/history_cnn_balanced.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✔ History guardado")


# Gráficos de Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Balanced - Accuracy vs Epochs')
plt.legend()
plt.tight_layout()
path_acc = f"{output_dir}/grafico_accuracy_cnn_balanced2.png"
plt.savefig(path_acc, dpi=300)
plt.close()
print("✔ Guardado:", os.path.abspath(path_acc))

# Gráfico Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN Balanced - Loss vs Epochs')
plt.legend()
plt.tight_layout()
path_loss = f"{output_dir}/grafico_loss_cnn_balanced2.png"
plt.savefig(path_loss, dpi=300)
plt.close()
print("✔ Guardado:", os.path.abspath(path_loss))


# Matriz de Confusión
print("\nGenerando Matriz de Confusión...")

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

preds = model.predict(test_ds)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
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
plt.title("Matriz de Confusión - CNN Balanceada")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

path_cm = f"{output_dir}/matriz_confusion_cnn_balanced2.png"
plt.savefig(path_cm, dpi=300)
plt.close()
print("✔ Guardado:", os.path.abspath(path_cm))

assert os.path.exists(path_acc)
assert os.path.exists(path_loss)
assert os.path.exists(path_cm)

print("\nReporte de Clasificación CNN (Pesos Suavizados):")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

print("\nTODO GUARDADO CORRECTAMENTE EN:", os.path.abspath(output_dir))