
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import keras_tuner as kt

# data
print("Cargando dataset...")
df = pd.read_csv("dataset.csv")

X = df.drop(columns=['Severity'])
y = df['Severity']
 
y = y - 1

print("Dividiendo datos...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.0101,
    random_state=42,
    stratify=y_train_full
)

def build_model(hp):
    model = keras.Sequential()
    
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))

    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', 64, 256, step=64),
            activation='relu'
        ))

        # Dropout
        if hp.Boolean(f'dropout_{i}'):
            model.add(keras.layers.Dropout(
                rate=hp.Float(f'dropout_rate_{i}', 0.2, 0.4, step=0.1)
            ))

    # Capa de salida 
    model.add(keras.layers.Dense(4, activation='softmax'))

    # Learning rate
    lr = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=50,  
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='dnn_multiclass'
)                               


# callback
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)


tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=2048,
    callbacks=[early_stop]
)

best_model = tuner.get_best_models(num_models=1)[0]


#Entrenar el mejor modelo encontrado por AutoML
history_automl = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=2048,
    callbacks=[early_stop]
)


# Evaluar
loss, acc = best_model.evaluate(X_test, y_test)
print(f"Precisión final: {acc*100:.2f}%")

model_automl = best_model

#---------gráficas---------
hist = history_automl.history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist['accuracy'], label='Entrenamiento')
plt.plot(hist['val_accuracy'], label='Validación')
plt.title('Precisión AutoML')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(hist['loss'], label='Entrenamiento')
plt.plot(hist['val_loss'], label='Validación')
plt.title('Pérdida AutoML')
plt.legend()
plt.show()

# Evaluación modelo final con datos de test 
loss, acc = model_automl.evaluate(X_test, y_test)
print(f"\n✅ Precisión final: {acc*100:.2f}%")

