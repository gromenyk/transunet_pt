import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from PIL import Image

npz_folder = '../data/Synapse/test_vol_h5'
predicted_images_folder = './predicted_images'
placed_centroids = './placed_center_of_mass'
centroid_over_pred_image = './center_of_mass_over_pred_images'
os.makedirs(placed_centroids, exist_ok=True)
os.makedirs(centroid_over_pred_image, exist_ok=True)

original_size = (512, 512)
prediction_size = (224, 224)

scale_x = original_size[1] / prediction_size[1]
scale_y = original_size[0] / prediction_size[0]

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
    if npz_file.endswith('npz'):
        data = np.load(os.path.join(npz_folder, npz_file))
        image = data['image']

    prediction_path = os.path.join(predicted_images_folder, npz_file.replace('.npz','_prediction.png'))
    if not os.path.exists(prediction_path):
        print(f'No prediction found for {npz_file}')
        continue

    prediction_image = Image.open(prediction_path).convert('L')
    prediction = np.array(prediction_image, dtype=np.float32) / 255.0

    left_center, right_center = find_centers_of_mass_for_hottest_pixels(prediction)

    scaled_left = (int(left_center[0] * scale_y), int(left_center[1] * scale_x))
    scaled_right = (int(right_center[0] * scale_y), int(right_center[1] * scale_x))

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.scatter(scaled_left[1], scaled_left[0], c='green', label='Left Center', s=20)
    ax.scatter(scaled_right[1], scaled_right[0], c='blue', label='Right Center', s=20)
    ax.set_title('Insertion Predictions')

    output_path = os.path.join(placed_centroids, npz_file.replace('npz', '_with_centroids.png'))
    plt.savefig(output_path)
    plt.close()

    fig, ax = plt.subplots()
    ax.imshow(prediction, cmap='hot')
    ax.scatter(left_center[1], left_center[0], c='green', label='Left Center of Mass', s=20)
    ax.scatter(right_center[1], right_center[0], c='blue', label='Right Center of Mass', s=20)
    ax.axvline(x=prediction.shape[1]//2, color='white', linestyle='--', label='Midline')
    ax.set_title('Centers of Mass on Heatmap')

    output_path_predicted = os.path.join(centroid_over_pred_image, npz_file.replace('npz','_with_centroids_predicted.png'))
    plt.savefig(output_path_predicted)
    plt.close()

    print(f'Process finished: {output_path}')