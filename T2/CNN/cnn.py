import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers

# Carga de datos

#PRUEBA CON OVERSAMPLIG 
#from balance_clases import cargar_datasets_balanceados

base_path = "datos_particionados"  

# Invocamos la función. Ya nos entrega TODAS las variables necesarias y limpias.
'''
train_ds, val_ds, test_ds, class_names = cargar_datasets_balanceados(
    base_path=base_path,
    image_size=(128, 128),
    batch_size=32,
    umbral_minimo=300
)
'''

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


'''
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# PRUEBA APLICANDO CLASS WEIGHTS 
class_weights = dict(enumerate(class_weights))

print("\nClass weights:")
for k, v in class_weights.items():
    print(f"Clase {k} ({class_names[k]}): {v:.2f}")

'''

# Configuración de salida
output_dir = "resultados_cnn"
os.makedirs(output_dir, exist_ok=True)
print("Guardando resultados en:", os.path.abspath(output_dir))



# Modelo CNN simple desde cero
model = Sequential([

    layers.Input(shape=(128,128,3)),

    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    #layers.Dropout(0.5),

    Dense(23, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



# Entrenamiento
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    #class_weight=class_weights
)

end_time = time.time()

training_time = end_time - start_time
print(f"\nTiempo entrenamiento: {training_time/60:.2f} minutos")



# Evaluación en test
test_loss, test_acc = model.evaluate(test_ds)

print(f"\n[RESULTADOS CNN]")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Guardar Modelo
model.save(f"{output_dir}/cnn_model.keras")
print("✔ Modelo guardado")


# Guardar History
with open(f"{output_dir}/history_cnn.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✔ History guardado")

# Gráfico Acuracy
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN - Accuracy vs Epochs')
plt.legend()
plt.tight_layout()

path_acc = f"{output_dir}/grafico_accuracy_cnn_dropout05.png"
plt.savefig(path_acc, dpi=300)
plt.close()

print("✔ Guardado:", os.path.abspath(path_acc))

# Loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN - Loss vs Epochs')
plt.legend()
plt.tight_layout()

path_loss = f"{output_dir}/grafico_loss_cnn_dropout05.png"
plt.savefig(path_loss, dpi=300)
plt.close()

print("✔ Guardado:", os.path.abspath(path_loss))

# Matriz de Confusión
print("\nGenerando Matriz de Confusión...")

# etiquetas reales
y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# predicciones
preds = model.predict(test_ds)
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
plt.title("Matriz de Confusión - CNN")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

path_cm = f"{output_dir}/matriz_confusion_cnn_dropout05.png"
plt.savefig(path_cm, dpi=300)
plt.close()

print("✔ Guardado:", os.path.abspath(path_cm))

assert os.path.exists(path_acc)
assert os.path.exists(path_loss)
assert os.path.exists(path_cm)

# Reporte de clasificación (por clase)
print("\nReporte de Clasificación CNN:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

print("\nTODO GUARDADO CORRECTAMENTE EN:", os.path.abspath(output_dir))