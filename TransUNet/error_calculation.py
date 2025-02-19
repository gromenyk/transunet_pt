import os
import numpy as np
from scipy.ndimage import center_of_mass
from PIL import Image
import matplotlib.pyplot as plt
import wandb  # <-- Agregado

# Inicializa wandb
wandb.init(project="deeppatella", entity="deeppatella", name="mask_vs_pred_visualization")

# Rutas de carpetas
npz_folder = '../data/Synapse/test_vol_h5'
predictions_folder = './predicted_images'
visualizations_folder = './visualizations'
os.makedirs(visualizations_folder, exist_ok=True)

# Variables para almacenar las diferencias
distal_differences = []
proximal_differences = []

# Factores de escalado entre las predicciones (224x224) y las máscaras (512x512)
prediction_size = (224, 224)
mask_size = (512, 512)
scale_y = mask_size[0] / prediction_size[0]  
scale_x = mask_size[1] / prediction_size[1]  

def calculate_center_from_mask(mask):
    """Calcula el centro de masa de una máscara con un círculo blanco."""
    if np.sum(mask) == 0:  
        return None
    return center_of_mass(mask)

def calculate_pixel_difference(true_coords, pred_coords):
    """Calcula la distancia en píxeles entre las coordenadas verdaderas y predichas."""
    return np.sqrt((true_coords[0] - pred_coords[0])**2 + (true_coords[1] - pred_coords[1])**2)

def find_centers_of_mass_for_hottest_pixels(prediction):
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

for npz_file in os.listdir(npz_folder):
    if npz_file.endswith('.npz'):
        data = np.load(os.path.join(npz_folder, npz_file))
        label = data['label']  

        h, w = label.shape
        mid = w // 2

        left_mask = label[:, :mid]  
        right_mask = label[:, mid:]  

        true_distal_center = calculate_center_from_mask(left_mask)
        true_proximal_center = calculate_center_from_mask(right_mask)

        if true_distal_center is None or true_proximal_center is None:
            print(f"Máscara vacía en {npz_file}, saltando...")
            continue

        true_proximal_center = (true_proximal_center[0], true_proximal_center[1] + mid)

        prediction_path = os.path.join(predictions_folder, npz_file.replace('.npz', '_prediction.png'))
        if not os.path.exists(prediction_path):
            print(f"Predicción no encontrada para {npz_file}")
            continue

        prediction_image = Image.open(prediction_path).convert('L')
        prediction = np.array(prediction_image, dtype=np.float32) / 255.0

        pred_distal_center, pred_proximal_center = find_centers_of_mass_for_hottest_pixels(prediction)

        scaled_pred_distal = (pred_distal_center[0] * scale_y, pred_distal_center[1] * scale_x)
        scaled_pred_proximal = (pred_proximal_center[0] * scale_y, pred_proximal_center[1] * scale_x)

        fig, ax = plt.subplots()
        ax.imshow(label, cmap='gray')
        ax.scatter(true_distal_center[1], true_distal_center[0], color='green', label='True Distal', s=50)
        ax.scatter(true_proximal_center[1], true_proximal_center[0], color='blue', label='True Proximal', s=50)
        ax.scatter(scaled_pred_distal[1], scaled_pred_distal[0], color='yellow', label='Pred Distal', s=50)
        ax.scatter(scaled_pred_proximal[1], scaled_pred_proximal[0], color='red', label='Pred Proximal', s=50)
        ax.legend()
        ax.set_title(f"Visualización para {npz_file}")

        output_image_path = os.path.join(visualizations_folder, npz_file.replace(".npz", "_visualization.png"))
        plt.savefig(output_image_path)
        plt.close()
        print(f"Visualización guardada en: {output_image_path}")

        # ✅ Registrar la imagen en wandb
        wandb.log({
            "Mask vs Prediction Visualization": wandb.Image(output_image_path, caption=npz_file)
        })

        print(f"Processing {npz_file}")

        distal_difference = calculate_pixel_difference(true_distal_center, scaled_pred_distal)
        proximal_difference = calculate_pixel_difference(true_proximal_center, scaled_pred_proximal)

        distal_differences.append(distal_difference)
        proximal_differences.append(proximal_difference)

if distal_differences:
    mean_distal_difference = np.mean(distal_differences)
    print(f"\nAverage error for distal insertion: {mean_distal_difference:.2f} píxeles")
    wandb.log({"Mean Distal Error (px)": mean_distal_difference})  

if proximal_differences:
    mean_proximal_difference = np.mean(proximal_differences)
    print(f"Average error for proximal insertion: {mean_proximal_difference:.2f} píxeles")
    wandb.log({"Mean Proximal Error (px)": mean_proximal_difference})  

wandb.finish()  
