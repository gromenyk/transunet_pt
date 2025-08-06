import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from PIL import Image
import pandas as pd
import csv
from filterpy.kalman import KalmanFilter
import cv2
import shutil
import glob

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

def run_postprocessing_pipeline(cfg):
    npz_folder = cfg['npz_folder']
    predicted_images_folder = cfg['predicted_images_folder']
    placed_centroids = cfg['placed_centroids']
    centroid_over_pred_image = cfg['centroid_over_pred_image']
    csv_file_path = cfg['csv_output_path']
    original_size = tuple(cfg.get('original_size', [512, 512]))
    prediction_size = tuple(cfg.get('prediction_size', [224, 224]))

    os.makedirs(placed_centroids, exist_ok=True)
    os.makedirs(centroid_over_pred_image, exist_ok=True)

    scale_x = original_size[1] / prediction_size[1]
    scale_y = original_size[0] / prediction_size[0]

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['npz_file', 'left_x', 'left_y', 'right_x', 'right_y'])

        for npz_file in os.listdir(npz_folder):
            if not npz_file.endswith('npz'):
                continue

            data = np.load(os.path.join(npz_folder, npz_file))
            image = data['image']

            prediction_path = os.path.join(predicted_images_folder, npz_file.replace('.npz','_prediction.png'))
            if not os.path.exists(prediction_path):
                print(f'❌ No prediction found for {npz_file}')
                continue

            prediction_image = Image.open(prediction_path).convert('L')
            prediction = np.array(prediction_image, dtype=np.float32) / 255.0

            left_center, right_center = find_centers_of_mass_for_hottest_pixels(prediction)
            scaled_left = (left_center[0] * scale_y, left_center[1] * scale_x)
            scaled_right = (right_center[0] * scale_y, right_center[1] * scale_x)

            writer.writerow([npz_file, scaled_left[0], scaled_left[1], scaled_right[0], scaled_right[1]])

            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')
            ax.scatter(scaled_left[1], scaled_left[0], c='green', s=20)
            ax.scatter(scaled_right[1], scaled_right[0], c='blue', s=20)
            ax.set_title('Insertion Predictions')
            output_path = os.path.join(placed_centroids, npz_file.replace('.npz', '_with_centroids.png'))
            plt.savefig(output_path)
            plt.close()

            fig, ax = plt.subplots()
            ax.imshow(prediction, cmap='hot')
            ax.scatter(left_center[1], left_center[0], c='yellow', s=20)
            ax.scatter(right_center[1], right_center[0], c='red', s=20)
            ax.axvline(x=prediction.shape[1]//2, color='white', linestyle='--')
            ax.set_title('Centers of Mass on Heatmap')
            output_pred_path = os.path.join(centroid_over_pred_image, npz_file.replace('.npz', '_with_centroids_predicted.png'))
            plt.savefig(output_pred_path)
            plt.close()

            print(f'Process finished for npz file: {npz_file}')

def calculate_prediction_errors(cfg):
    npz_folder = cfg['npz_folder']
    predictions_folder = cfg['predicted_images_folder']
    visualizations_folder = cfg['visualizations_folder']
    csv_path = cfg['csv_output_path']
    original_size = tuple(cfg.get('original_size', [512, 512]))
    prediction_size = tuple(cfg.get('prediction_size', [224, 224]))

    os.makedirs(visualizations_folder, exist_ok=True)

    scale_y = original_size[0] / prediction_size[0]
    scale_x = original_size[1] / prediction_size[1]

    distal_differences = []
    proximal_differences = []
    true_coords = []

    for npz_file in os.listdir(npz_folder):
        if not npz_file.endswith('.npz'):
            continue

        npz_path = os.path.join(npz_folder, npz_file)
        pred_path = os.path.join(predictions_folder, npz_file.replace('.npz', '_prediction.png'))

        if not os.path.exists(pred_path):
            print(f"❌ Falta predicción para: {npz_file}")
            continue

        data = np.load(npz_path)
        label = data['label']
        h, w = label.shape
        mid = w // 2
        left_mask = label[:, :mid]
        right_mask = label[:, mid:]

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            print(f"⚠️ Máscara vacía: {npz_file}")
            continue

        true_distal = center_of_mass(left_mask)
        true_proximal = center_of_mass(right_mask)
        true_proximal = (true_proximal[0], true_proximal[1] + mid)

        prediction = np.array(Image.open(pred_path).convert('L'), dtype=np.float32) / 255.0

        pred_distal, pred_proximal = find_centers_of_mass_for_hottest_pixels(prediction)
        scaled_pred_distal = (pred_distal[0] * scale_y, pred_distal[1] * scale_x)
        scaled_pred_proximal = (pred_proximal[0] * scale_y, pred_proximal[1] * scale_x)

        fig, ax = plt.subplots()
        ax.imshow(label, cmap='gray')
        ax.scatter(true_distal[1], true_distal[0], color='green', label='True Distal')
        ax.scatter(true_proximal[1], true_proximal[0], color='blue', label='True Proximal')
        ax.scatter(scaled_pred_distal[1], scaled_pred_distal[0], color='yellow', label='Pred Distal')
        ax.scatter(scaled_pred_proximal[1], scaled_pred_proximal[0], color='red', label='Pred Proximal')
        ax.legend()
        ax.set_title(f"Errors: {npz_file}")
        vis_path = os.path.join(visualizations_folder, npz_file.replace(".npz", "_vis.png"))
        plt.savefig(vis_path)
        plt.close()

        err_distal = np.sqrt((true_distal[0] - scaled_pred_distal[0])**2 + (true_distal[1] - scaled_pred_distal[1])**2)
        err_proximal = np.sqrt((true_proximal[0] - scaled_pred_proximal[0])**2 + (true_proximal[1] - scaled_pred_proximal[1])**2)

        distal_differences.append(err_distal)
        proximal_differences.append(err_proximal)

        true_coords.append({
            'File': npz_file,
            'True Distal X': true_distal[0],
            'True Distal Y': true_distal[1],
            'True Proximal X': true_proximal[0],
            'True Proximal Y': true_proximal[1]
        })

    pd.DataFrame(true_coords).to_csv("true_coords.csv", index=False)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distal Errors', 'Proximal Errors'])
        for d, p in zip(distal_differences, proximal_differences):
            writer.writerow([d, p])

    print(f"Errors csv saved in: {csv_path}")
    print(f"Mean error distal: {np.mean(distal_differences):.2f}px")
    print(f"Mean error proximal: {np.mean(proximal_differences):.2f}px")

def calculate_acceleration_from_predictions(cfg):
    input_csv_path = cfg["input_csv_path"]
    output_csv_path = cfg["output_csv_path"]
    delta_t = 1 / cfg.get("frame_rate", 51)  # FPS

    if not os.path.exists(input_csv_path):
        print(f"File not found: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)

    distances_distal_x = [0]
    distances_distal_y = [0]
    distances_distal = [0]
    speed_distal_x = [0]
    speed_distal_y = [0]
    speed_distal = [0]
    acceleration_distal = [0]
    distances_proximal_x = [0]
    distances_proximal_y = [0]
    distances_proximal = [0]
    speed_proximal_x = [0]
    speed_proximal_y = [0]
    speed_proximal = [0]
    acceleration_proximal = [0]

    for i in range(1, len(df)):
        dist_d = np.hypot(df.loc[i, 'left_x'] - df.loc[i-1, 'left_x'], df.loc[i, 'left_y'] - df.loc[i-1, 'left_y'])
        dist_p = np.hypot(df.loc[i, 'right_x'] - df.loc[i-1, 'right_x'], df.loc[i, 'right_y'] - df.loc[i-1, 'right_y'])

        distances_distal.append(dist_d)
        distances_proximal.append(dist_p)

        speed_d = dist_d / delta_t
        speed_p = dist_p / delta_t

        accel_d = abs(speed_d - speed_distal[i-1]) / delta_t
        accel_p = abs(speed_p - speed_proximal[i-1]) / delta_t

        speed_distal.append(speed_d)
        acceleration_distal.append(accel_d)
        speed_proximal.append(speed_p)
        acceleration_proximal.append(accel_p)

        dx_d = df.loc[i, 'left_x'] - df.loc[i-1, 'left_x']
        dy_d = df.loc[i, 'left_y'] - df.loc[i-1, 'left_y']
        dx_p = df.loc[i, 'right_x'] - df.loc[i-1, 'right_x']
        dy_p = df.loc[i, 'right_y'] - df.loc[i-1, 'right_y']

        distances_distal_x.append(dx_d)
        distances_distal_y.append(dy_d)
        distances_proximal_x.append(dx_p)
        distances_proximal_y.append(dy_p)

        speed_distal_x.append(dx_d / delta_t)
        speed_distal_y.append(dy_d / delta_t)
        speed_proximal_x.append(dx_p / delta_t)
        speed_proximal_y.append(dy_p / delta_t)

    df['distances_distal_x'] = distances_distal_x
    df['distances_distal_y'] = distances_distal_y
    df['distal_distance'] = distances_distal
    df['speed_distal_x'] = speed_distal_x
    df['speed_distal_y'] = speed_distal_y
    df['distal_acceleration'] = acceleration_distal
    df['distances_proximal_x'] = distances_proximal_x
    df['distances_proximal_y'] = distances_proximal_y
    df['proximal_distance'] = distances_proximal
    df['speed_proximal_x'] = speed_proximal_x
    df['speed_proximal_y'] = speed_proximal_y
    df['proximal_acceleration'] = acceleration_proximal

    df.to_csv(output_csv_path, index=False)
    print(f"Accelerations saved in: {output_csv_path}")


def apply_kalman_filter_distal(cfg):
    input_csv_path = cfg["input_csv_path"]
    output_csv_path = cfg["output_csv_path"]
    output_plot_path = cfg["output_plot_path"]
    umbral_acceleration = cfg.get("threshold_acceleration", 0)
    fps = cfg.get("frame_rate", 51)
    delta_t = 1 / fps

    if not os.path.exists(input_csv_path):
        print(f"File not found: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([
        df['left_x'].iloc[0],
        df['speed_distal_x'].iloc[0],
        df['left_y'].iloc[0],
        df['speed_distal_y'].iloc[0]
    ])

    kf.F = np.array([
        [1, delta_t, 0, 0],
        [0, 1,       0, 0],
        [0, 0,       1, delta_t],
        [0, 0,       0, 1]
    ])

    kf.P = np.eye(4) * 0.1
    kf.Q = np.eye(4) * 0.1
    kf.R = np.eye(2) * 1.0
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    predictions = []
    real_x = []
    real_y = []

    for i, row in df.iterrows():
        accel = row['distal_acceleration']
        real_x.append(row['left_x'])
        real_y.append(row['left_y'])

        kf.predict()

        if accel >= umbral_acceleration:
            z = np.array([row['left_x'], row['left_y']])
            kf.update(z)
            predictions.append([kf.x[0], kf.x[2]])
        else:
            predictions.append([row['left_x'], row['left_y']])

    pred_df = pd.DataFrame(predictions, columns=['Kalman_Predicted_Distal_X', 'Kalman_Predicted_Distal_Y'])
    df = pd.concat([df, pred_df], axis=1)

    df.to_csv(output_csv_path, index=False)
    print(f"Coordinates with applied Kalman filter saved in: {output_csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, real_x, label='X original (TransUNet)', color='blue')
    plt.plot(df.index, real_y, label='Y original (TransUNet)', color='green')
    plt.plot(df.index, pred_df['Kalman_Predicted_Distal_X'], label='X Kalman', color='red', linestyle='--')
    plt.plot(df.index, pred_df['Kalman_Predicted_Distal_Y'], label='Y Kalman', color='orange', linestyle='--')
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Position")
    plt.title("Kalman Filter Distal Position Prediction")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()

    print(f"Plot guardado en: {output_plot_path}")

def apply_kalman_filter_proximal(cfg):
    input_csv_path = cfg["input_csv_path"]
    output_csv_path = cfg["output_csv_path"]
    output_plot_path = cfg["output_plot_path"]
    umbral_acceleration = cfg.get("threshold_acceleration", 0)
    fps = cfg.get("frame_rate", 51)
    delta_t = 1 / fps

    if not os.path.exists(input_csv_path):
        print(f"File not found: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([
        df['right_x'].iloc[0],
        df['speed_proximal_x'].iloc[0],
        df['right_y'].iloc[0],
        df['speed_proximal_y'].iloc[0]
    ])

    kf.F = np.array([
        [1, delta_t, 0, 0],
        [0, 1,       0, 0],
        [0, 0,       1, delta_t],
        [0, 0,       0, 1]
    ])

    kf.P = np.eye(4) * 0.1
    kf.Q = np.eye(4) * 0.1
    kf.R = np.eye(2)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    predictions = []
    real_x = []
    real_y = []

    for _, row in df.iterrows():
        accel = row['proximal_acceleration']
        real_x.append(row['right_x'])
        real_y.append(row['right_y'])

        kf.predict()

        if accel >= umbral_acceleration:
            z = np.array([row['right_x'], row['right_y']])
            kf.update(z)
            predictions.append([kf.x[0], kf.x[2]])
        else:
            predictions.append([row['right_x'], row['right_y']])

    pred_df = pd.DataFrame(predictions, columns=['Kalman_Predicted_Proximal_X', 'Kalman_Predicted_Proximal_Y'])
    df = pd.concat([df, pred_df], axis=1)

    df.to_csv(output_csv_path, index=False)
    print(f"Coordinates with applied Kalman filter saved in: {output_csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, real_x, label='X original (TransUNet)', color='blue')
    plt.plot(df.index, real_y, label='Y original (TransUNet)', color='green')
    plt.plot(df.index, pred_df['Kalman_Predicted_Proximal_X'], label='X Kalman', color='red', linestyle='--')
    plt.plot(df.index, pred_df['Kalman_Predicted_Proximal_Y'], label='Y Kalman', color='orange', linestyle='--')
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Position")
    plt.title("Kalman Filter Applied to Proximal Coordinates")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()

    print(f"Plot saved in: {output_plot_path}")

def reconstruct_video_from_frames(cfg):
    frames_folder = cfg["frames_folder"]
    output_video_file = cfg["output_video_file"]
    frame_rate = cfg.get("frame_rate", 51)

    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])

    if not frame_files:
        print(f"No frames found in {frames_folder}")
        return

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Reconstructed video saved in: {output_video_file}")

def safe_delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"🗑️ Carpeta eliminada: {folder_path}")
    else:
        print(f"📁 No encontrada (skip): {folder_path}")

def safe_delete_files(pattern):
    files = glob.glob(pattern)
    if files:
        for f in files:
            os.remove(f)
            print(f"🗑️ Archivo eliminado: {f}")
    else:
        print(f"📄 No se encontraron archivos para: {pattern}")

def clean_pipeline_outputs(cfg):
    import shutil
    import glob

    folders_to_remove = cfg.get("folders", [])
    npz_patterns = cfg.get("npz_patterns", [])
    file_patterns = cfg.get("file_patterns", [])

    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Eliminated folder: {folder}")
        else:
            print(f"Folder not found (skip): {folder}")

    for pattern in npz_patterns + file_patterns:
        files = glob.glob(pattern)
        if files:
            for f in files:
                os.remove(f)
                print(f"Eliminated file: {f}")
        else:
            print(f"No files found for: {pattern}")