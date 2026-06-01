import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold


base_path = "datos_particionados"  

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/test",
    image_size=(128, 128),
    batch_size=32,
    shuffle=False,
    crop_to_aspect_ratio=True
)

# Cargamos por separado train y val para unirlos en memoria para el K-Fold
train_ds_temp = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/train", image_size=(128, 128), batch_size=32, shuffle=False, crop_to_aspect_ratio=True
)
val_ds_temp = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/val", image_size=(128, 128), batch_size=32, shuffle=False, crop_to_aspect_ratio=True
)

class_names = train_ds_temp.class_names
num_clases = len(class_names)  
print(f"Dataset listo. Detectadas {num_clases} clases de peces.")


# Preparación de datos en memoria para K-Fold
print("\nPreparando datos en memoria para K-Fold...")

def extract_images_and_labels(dataset):
    images, labels = [], []
    for imgs, lbls in dataset:
        images.append(imgs.numpy())
        labels.append(lbls.numpy())
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

X_train_part, y_train_part = extract_images_and_labels(train_ds_temp)
X_val_part, y_val_part = extract_images_and_labels(val_ds_temp)

# Juntamos train y val en un único conjunto de desarrollo
X_dev = np.concatenate([X_train_part, X_val_part], axis=0)
y_dev = np.concatenate([y_train_part, y_val_part], axis=0)

print(f"Total de imágenes para Validación Cruzada: {X_dev.shape[0]}")

output_dir = "resultados_cnn"
os.makedirs(output_dir, exist_ok=True)


def crear_modelo():
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
        Dropout(0.5), # Reactivado para evitar Overfitting

        Dense(num_clases, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


output_dir = "resultados_cnn"
os.makedirs(output_dir, exist_ok=True)

# Configuración de K-Fold Cross-Validation
k_folds = 3  
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

histories = []
scores_por_fold = []

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
    print(f"\n=== ENTRENANDO EN EL FOLD {fold + 1} DE {k_folds} ===")
    
    X_train, y_train = X_dev[train_idx], y_dev[train_idx]
    X_val, y_val = X_dev[val_idx], y_dev[val_idx]
    
    # Los pesos se calculan dentro de cada fold usando únicamente los datos de entrenamiento de ese fold
    classes_fold = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes_fold, y=y_train)
    class_weights = dict(zip(classes_fold, weights))
    
    # Crear modelo limpio
    model = crear_modelo()
    
    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        class_weight=class_weights, # Activado para balancear tus clases pequeñas
        verbose=1
    )
    
    # Evaluar con los datos de validación de este fold
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold + 1} terminado. Val Accuracy: {val_acc:.4f}")
    
    scores_por_fold.append(val_acc)
    histories.append(history.history)

end_time = time.time()
print(f"\nTiempo total de entrenamiento ({k_folds} folds): {(end_time - start_time)/60:.2f} minutos")
print(f"Precisión media de validación cruzada: {np.mean(scores_por_fold):.4f} (+/- {np.std(scores_por_fold):.4f})")


#Evaluacion con test 
print("\n=== EVALUACIÓN FINAL EN CONJUNTO DE TEST FIJO ===")
# Evaluamos usando el último modelo entrenado
test_loss, test_acc = model.evaluate(test_ds)

print(f"\n[RESULTADOS FINALES CNN]")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Guardar último modelo entrenado
model.save(f"{output_dir}/cnn_model_kfold.keras")
print("✔ Modelo guardado")

# Guardar historiales
with open(f"{output_dir}/history_cnn_kfold.pkl", "wb") as f:
    pickle.dump(histories, f)


plt.figure(figsize=(12, 5))

# Gráfico de Accuracy
plt.subplot(1, 2, 1)
for i, h in enumerate(histories):
    plt.plot(h['accuracy'], label=f'Train Fold {i+1}', linestyle='--')
    plt.plot(h['val_accuracy'], label=f'Val Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('K-Fold - Accuracy')
plt.legend(fontsize='small')

# Gráfico de Loss
plt.subplot(1, 2, 2)
for i, h in enumerate(histories):
    plt.plot(h['loss'], label=f'Train Fold {i+1}', linestyle='--')
    plt.plot(h['val_loss'], label=f'Val Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('K-Fold - Loss')
plt.legend(fontsize='small')

plt.tight_layout()
path_graphs = f"{output_dir}/graficos_kfold.png"
plt.savefig(path_graphs, dpi=300)
plt.close()
print("✔ Gráficos de entrenamiento guardados")

print("\nGenerando Matriz de Confusión sobre Test...")

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

preds = model.predict(test_ds)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión Final (K-Fold + Class Weights)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

path_cm = f"{output_dir}/matriz_confusion_kfold.png"
plt.savefig(path_cm, dpi=300)
plt.close()
print("✔ Matriz de confusión guardada")

print("\nReporte de Clasificación Final CNN (K-Fold):")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("\nPROCESO COMPLETO. TODO GUARDADO EN:", os.path.abspath(output_dir))