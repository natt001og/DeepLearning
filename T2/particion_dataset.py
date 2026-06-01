import zipfile
import os
import splitfolders

zip_path = "fish_image.zip"
extract_path = "fish_image"
output_path = "datos_particionados"

# 2. Descomprimir el archivo ZIP localmente
print("Descomprimiendo el archivo ZIP...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

ruta_clases_originales = os.path.join(extract_path, "fish_image")

print("\nConteo de archivos extraídos:")
for root, dirs, files in os.walk(ruta_clases_originales):
    if len(files) > 0:
        print(f"Encontrados {len(files)} archivos en: {root}")

print("\nParticionando el dataset ")
splitfolders.ratio(
    ruta_clases_originales, 
    output=output_path, 
    seed=42,             # Semilla fija para que la partición sea reproducible
    ratio=(.7, .2, .1),  # Train=0.7, Val=0.2, Test=0.1
    group_prefix=None
)

print(f"\n¡Proceso terminado, las carpetas fijas están en: '{output_path}'")