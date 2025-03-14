import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from PIL import Image

NPZ_FOLDER = './data/Synapse/test_vol_h5'
PREDICTED_IMAGES_FOLDER = './predicted_images'
PLACED_CENTROIDS = './outputs/placed_center_of_mass'
CENTROID_OVER_PRED_IMAGE = './center_of_mass_over_pred_images'
ORIGINAL_IMAGES_FILE = '../data/Synapse/original_images.npy'

os.makedirs(PLACED_CENTROIDS, exist_ok=True)
os.makedirs(CENTROID_OVER_PRED_IMAGE, exist_ok=True)

def find_centers_of_mass_for_hottest_pixels(prediction):
    """Encuentra los centros de masa en las mitades izquierda y derecha de la imagen de predicción."""
    h, w = prediction.shape
    mid = w // 2

    left_half = prediction[:, :mid]
    right_half = prediction[:, mid:]

    left_max_value = np.max(left_half)
    right_max_value = np.max(right_half)

    left_mask = (left_half == left_max_value).astype(np.float32)
    right_mask = (right_half == right_max_value).astype(np.float32)

    left_com = center_of_mass(left_mask)
    right_com = center_of_mass(right_mask)

    left_center = (left_com[0], left_com[1])
    right_center = (right_com[0], right_com[1] + mid)

    return left_center, right_center

def hampel_filter(data, window_size=5, threshold=3):
    """Aplica el filtro Hampel para detectar y reemplazar outliers"""
    filtered_data = data.copy()
    half_window = window_size // 2
    
    for i in range(half_window, len(data) - half_window):
        window = data[i - half_window:i + half_window + 1]
        median = np.median(window)  
        mad = np.median(np.abs(window - median))  
        
        threshold_value = threshold * mad
        
        if np.abs(data[i] - median) > threshold_value:
            filtered_data[i] = median
    
    return filtered_data

def smooth_coordinates_hampel(center_list, window_size=5, threshold=3):
    """Aplica el filtro Hampel para suavizar las coordenadas de los centros."""
    smoothed_list = []
    for center in center_list:
        smoothed_x = hampel_filter([coord[0] for coord in center_list], window_size, threshold)
        smoothed_y = hampel_filter([coord[1] for coord in center_list], window_size, threshold)
        smoothed_list.append((smoothed_x, smoothed_y))
    
    return smoothed_list

def place_centroids(npz_files, predictions_dir, original_images_file, placed_centroids_folder):
    """Procesa los archivos NPZ y genera imágenes con los centroides marcados."""

    ORIGINAL_SIZE = (512, 512)
    PREDICTION_SIZE = (224, 224)
    SCALE_X = ORIGINAL_SIZE[1] / PREDICTION_SIZE[1]
    SCALE_Y = ORIGINAL_SIZE[0] / PREDICTION_SIZE[0]

    original_images = np.load(original_images_file)

    os.makedirs(placed_centroids_folder, exist_ok=True)

    left_center_list = []
    right_center_list = []

    for npz_file in os.listdir(npz_files):
        if not npz_file.endswith('.npz'):
            continue

        data = np.load(os.path.join(npz_files, npz_file))
        image = data['image']

        prediction_path = os.path.join(predictions_dir, npz_file.replace('.npz', '_prediction.png'))
        if not os.path.exists(prediction_path):
            print(f'No prediction found for {npz_file}')
            continue

        prediction_image = Image.open(prediction_path).convert('L')
        prediction = np.array(prediction_image, dtype=np.float32) / 255.0

        left_center, right_center = find_centers_of_mass_for_hottest_pixels(prediction)

        scaled_left = (int(left_center[0] * SCALE_Y), int(left_center[1] * SCALE_X))
        scaled_right = (int(right_center[0] * SCALE_Y), int(right_center[1] * SCALE_X))

        left_center_list.append(scaled_left)
        right_center_list.append(scaled_right)

        frame_index = int(npz_file.split('_')[1].split('.')[0])
        original_image = original_images[frame_index]

        save_centroid_image(original_image, scaled_left, scaled_right, npz_file, placed_centroids_folder)


def save_centroid_image(original_image, scaled_left, scaled_right, npz_file, placed_centroids):
    """Guarda la imagen con los centroides sobre la imagen original."""
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap='gray')
    ax.scatter(scaled_left[1], scaled_left[0], c='green', label='Left Center', s=20)
    ax.scatter(scaled_right[1], scaled_right[0], c='blue', label='Right Center', s=20)
    ax.set_title('Insertion Predictions')

    output_path = os.path.join(placed_centroids, npz_file.replace('.npz', '_with_centroids.png'))
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    place_centroids(NPZ_FOLDER, PREDICTED_IMAGES_FOLDER, ORIGINAL_IMAGES_FILE, PLACED_CENTROIDS)



