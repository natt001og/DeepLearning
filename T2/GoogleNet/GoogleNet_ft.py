import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

output_dir = "resultados_googlenet_finetuning"
os.makedirs(output_dir, exist_ok=True)

preprocess_layer = tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input)

# Transfer learning: pesos preentrenados de ImageNet
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# FINE-TUNING: descongelar ultimas 30 capas
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Capas totales: {len(base_model.layers)}")
print(f"Capas entrenables (fine-tuning): {sum(layer.trainable for layer in base_model.layers)}")

model_inception = Sequential([
    preprocess_layer,
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_clases, activation='softmax')
])

model_inception.build(input_shape=(None, 128, 128, 3))
model_inception.summary()

model_inception.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando entrenamiento de GoogLeNet/Inception (FINE-TUNING - 50 capas descongeladas)...")
start_time = time.time()
history_inception = model_inception.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
end_time = time.time()
training_time = end_time - start_time
print(f"Tiempo entrenamiento: {training_time/60:.2f} minutos")

print("\nEvaluando GoogLeNet/Inception en Test...")
test_loss_inception, test_accuracy_inception = model_inception.evaluate(test_ds)
print(f"\n[RESULTADOS] Test Accuracy: {test_accuracy_inception:.4f} | Test Loss: {test_loss_inception:.4f}")

model_inception.save(f"{output_dir}/googlenet_finetuning.keras")
with open(f"{output_dir}/history_googlenet_finetuning.pkl", "wb") as f:
    pickle.dump(history_inception.history, f)

plt.figure(figsize=(6, 4))
plt.plot(history_inception.history['accuracy'], label='Train')
plt.plot(history_inception.history['val_accuracy'], label='Validation')
plt.title('GoogLeNet/Inception - FINE-TUNING (50 capas descongeladas)')
plt.ylabel('Accuracy')
plt.xlabel('Epocas')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_plot_finetuning.png')
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history_inception.history['loss'], label='Train')
plt.plot(history_inception.history['val_loss'], label='Validation')
plt.title('GoogLeNet/Inception - FINE-TUNING (50 capas descongeladas)')
plt.ylabel('Loss')
plt.xlabel('Epocas')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/loss_plot_finetuning.png')
plt.close()

print("\nGenerando matriz de confusion...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
preds = model_inception.predict(test_ds)
y_pred = np.argmax(preds, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusion - GoogLeNet/Inception FINE-TUNING")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix_finetuning.png')
plt.close()

print("\nReporte de Clasificacion:")
print(classification_report(y_true, y_pred, target_names=class_names))
print(f"\nTODO GUARDADO EN: {os.path.abspath(output_dir)}")