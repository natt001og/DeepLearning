import os
import glob
import tensorflow as tf

def cargar_datasets_balanceados(base_path, image_size=(128, 128), batch_size=32, umbral_minimo=300):
    """
    Carga los datasets aplicando oversampling condicional de forma rápida y limpia.
    Corrige las dimensiones de entrada sin usar buffers pesados que ralentizan la CPU.
    """
    print(f"\n[BALANCEADOR] Cargando datos rápidos desde: '{base_path}'")

    # 1. Carga base de directorios (Mezclado rápido nativo)
    raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{base_path}/train",
        image_size=image_size,
        batch_size=1,
        shuffle=True, 
        crop_to_aspect_ratio=True
    )
    class_names = raw_train_ds.class_names
    num_clases = len(class_names)  
    datasets_procesados = []

    # 2. Oversampling condicional directo
    for i in range(num_clases):
        nombre_clase = class_names[i]
        ruta_clase = os.path.join(base_path, "train", nombre_clase)
        
        ds_clase = raw_train_ds.filter(lambda x, y: tf.equal(y[0], i))
        num_fotos_reales = len(glob.glob(os.path.join(ruta_clase, "*")))

        if num_fotos_reales < umbral_minimo:
            ds_ajustado = ds_clase.repeat().take(umbral_minimo)
        else:
            ds_ajustado = ds_clase

        datasets_procesados.append(ds_ajustado)

    # 3. Reunir las clases en un solo flujo
    train_ds_balanced = datasets_procesados[0]
    for ds_siguiente in datasets_procesados[1:]:
        train_ds_balanced = train_ds_balanced.concatenate(ds_siguiente)

    # 4. Agrupación en batches limpia y corrección de dimensiones
    # .unbatch() limpia el tamaño individual y .batch(batch_size) arma los grupos de 32 sin demoras
    train_ds = (train_ds_balanced
                .unbatch()  
                .batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    # 5. Carga normal de Validación y Test (Idéntica a tu código original)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{base_path}/val",
        image_size=image_size,
        batch_size=batch_size,
        crop_to_aspect_ratio=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{base_path}/test",
        image_size=image_size,
        batch_size=batch_size,
        crop_to_aspect_ratio=True,
        shuffle=False
    )

    return train_ds, val_ds, test_ds, class_names