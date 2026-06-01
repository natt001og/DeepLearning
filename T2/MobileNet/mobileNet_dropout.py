import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
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

output_dir = "resultados_mobilenet_dropout_sin_weights"
os.makedirs(output_dir, exist_ok=True)

base_model = MobileNet(
    weights=None,
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = True

model_mobilenet = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_clases, activation='softmax')
])

model_mobilenet.build(input_shape=(None, 128, 128, 3))
model_mobilenet.summary()

model_mobilenet.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando entrenamiento de MobileNet CON DROPOUT (sin weights)...")
start_time = time.time()
history_mobilenet = model_mobilenet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
end_time = time.time()
training_time = end_time - start_time
print(f"Tiempo entrenamiento: {training_time/60:.2f} minutos")

print("\nEvaluando MobileNet en Test...")
test_loss_mobilenet, test_accuracy_mobilenet = model_mobilenet.evaluate(test_ds)
print(f"\n[RESULTADOS] Test Accuracy: {test_accuracy_mobilenet:.4f} | Test Loss: {test_loss_mobilenet:.4f}")

model_mobilenet.save(f"{output_dir}/mobilenet_dropout_sin_weights.keras")
with open(f"{output_dir}/history_mobilenet_dropout_sin_weights.pkl", "wb") as f:
    pickle.dump(history_mobilenet.history, f)

plt.figure(figsize=(6, 4))
plt.plot(history_mobilenet.history['accuracy'], label='Train')
plt.plot(history_mobilenet.history['val_accuracy'], label='Validation')
plt.title('MobileNet CON DROPOUT SIN WEIGHTS - Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epocas')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_plot.png')
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history_mobilenet.history['loss'], label='Train')
plt.plot(history_mobilenet.history['val_loss'], label='Validation')
plt.title('MobileNet CON DROPOUT SIN WEIGHTS - Loss')
plt.ylabel('Loss')
plt.xlabel('Epocas')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/loss_plot.png')
plt.close()

print("\nGenerando matriz de confusion...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
preds = model_mobilenet.predict(test_ds)
y_pred = np.argmax(preds, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusion - MobileNet CON DROPOUT SIN WEIGHTS")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png')
plt.close()

print("\nReporte de Clasificacion:")
print(classification_report(y_true, y_pred, target_names=class_names))
print(f"\nTODO GUARDADO EN: {os.path.abspath(output_dir)}")