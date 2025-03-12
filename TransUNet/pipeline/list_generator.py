import os
import argparse

# Path de la carpeta con los archivos .npz
npz_folder = './'  # Carpeta donde están los archivos .npz
output_txt_path = '../../../TransUNet/lists/lists_Synapse/test_vol.txt'  # Archivo de salida

#print(len(os.listdir(npz_folder)))


# Lista para almacenar los nombres de los archivos sin extensión
def npz_files_list(npz_folder, output_txt_path):
    file_names = []

    # Iterar por los archivos en la carpeta
    for file_name in os.listdir(npz_folder):
        if file_name.endswith('.npz'):  # Solo procesar archivos con extensión .npz
            # Remover la extensión del archivo
            base_name = os.path.splitext(file_name)[0]
            file_names.append(base_name)

    # Guardar los nombres en el archivo .txt
    with open(output_txt_path, 'w') as txt_file:
        txt_file.write('\n'.join(file_names))

    print(f"List generated in {output_txt_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate NPZ file list')
    parser.add_argument('--npz_folder', type=str, default='../../data/Synapse/test_vol_h5', help='Folder with the NPZ files')
    parser.add_argument('--output_txt_file', type=str, default='../lists/lists_Synapse/test_vol.txt', help='Output folder for the txt file')

    args = parser.parse_args()
    npz_files_list(args.npz_folder, args.output_txt_file)

if __name__ == '__main__':
    main()

