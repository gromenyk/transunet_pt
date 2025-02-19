import os
import numpy as np

npz_folder = "./"

# Verificar si la carpeta existe
if not os.path.exists(npz_folder):
    print(f"La carpeta '{npz_folder}' no existe.")
else:
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]

    for npz_file in npz_files:
        npz_path = os.path.join(npz_folder, npz_file)
        data = np.load(npz_path)

        if 'coords' not in data.files:
            print(f"El archivo {npz_file} no contiene 'coords'.")
