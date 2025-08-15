from scipy.ndimage import center_of_mass
import numpy as np
import pandas as pd
import json
import cv2
import csv
import os
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Find center of mass of the laft and right side of the predicted probability heatmap
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

# Scale the predicted coords to 632 x 508
def scale_predicted_coords(predictions_in_memory, target_size=(632, 508)):
    orig_h, orig_w = target_size
    scale_y = orig_h / 224
    scale_x = orig_w / 224

    for key, data in predictions_in_memory.items():
        pred_coords = data.get("predicted_coords", {})
        if not pred_coords:
            continue

        scaled_coords = {}
        for point_name, (y, x) in pred_coords.items():
            scaled_coords[point_name] = (
                y * scale_y,
                x * scale_x
            )

        data["predicted_coords_scaled"] = scaled_coords

    return predictions_in_memory

# Draw the predicted coordinates and the labeled ones on top of a frame (for visual confirmation)
def draw_gt_and_predictions(test_videos_dir, ndjson_path, predictions_in_memory, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(ndjson_path, "r", encoding="utf-8") as f:
        ndjson_data = [json.loads(line) for line in f]

    gt_by_video = {}
    for entry in ndjson_data:
        video_name = os.path.splitext(entry["data_row"]["external_id"])[0]
        project_id = next(iter(entry["projects"]))
        frames = entry["projects"][project_id]["labels"][0]["annotations"]["frames"]
        frame1_data = frames.get("1", {})
        gt_by_video[video_name] = frame1_data.get("objects", {})

    processed_videos_set = set()

    for pred_key, pred_data in predictions_in_memory.items():
        video_name = pred_key.split("_frame_")[0]

        if video_name in processed_videos_set:
            continue
        processed_videos_set.add(video_name)

        video_path = os.path.join(test_videos_dir, video_name + ".mp4")
        if not os.path.exists(video_path):
            print(f"Video not found for {video_name}")
            continue

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read first frame for {video_name}")
            continue

        # Draw predictions (red)
        pred_coords = pred_data.get("predicted_coords_scaled", {})
        for label, (y_pred, x_pred) in pred_coords.items():
            cv2.circle(frame, (int(x_pred), int(y_pred)), 5, (0, 0, 255), -1)

        # Draw labeled coordinates (green)
        if video_name in gt_by_video:
            for obj in gt_by_video[video_name].values():
                x = int(round(obj["point"]["x"]))
                y = int(round(obj["point"]["y"]))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Save image
        out_path = os.path.join(output_dir, f"{video_name}_gt_pred.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved to {out_path}")

# Load the labeled coordinates from the Labelbox NDJSON
def load_gt_coords(ndjson_path):
    with open(ndjson_path, "r", encoding="utf-8") as f:
        ndjson_data = [json.loads(line) for line in f]

    gt_coords_by_video = {}
    for entry in ndjson_data:
        video_name = os.path.splitext(entry["data_row"]["external_id"])[0]
        project_id = next(iter(entry["projects"]))
        frames = entry["projects"][project_id]["labels"][0]["annotations"]["frames"]

        for frame_number, frame_data in frames.items():
            objs = frame_data.get("objects", {})
            coords = {}
            for obj in objs.values():
                label = obj.get("value", "").lower()  
                x = obj["point"]["x"]
                y = obj["point"]["y"]
                coords[label] = (y, x)

            frame_key = f"{video_name}_frame_{int(frame_number)-1:04d}"
            gt_coords_by_video[frame_key] = coords

    return gt_coords_by_video


# Error calculation (as the Euclidean distance betweeen predicted coords and labeled coords)
def calculate_errors(predictions_in_memory, gt_coords_by_video, save_csv=True, csv_path="errors.csv"):
    rows = []

    for key, data in predictions_in_memory.items():
        video_frame_id = key
        pred_coords = data.get("predicted_coords_scaled", {})
        gt_coords = gt_coords_by_video.get(video_frame_id, {})

        row = {
            "video": video_frame_id,
            "gt_dtbi_x": None, "gt_dtbi_y": None,
            "gt_ptbi_x": None, "gt_ptbi_y": None,
            "pred_dtbi_x": None, "pred_dtbi_y": None,
            "pred_ptbi_x": None, "pred_ptbi_y": None,
            "Distal Errors": None,
            "Proximal Errors": None
        }

        if "dtbi" in gt_coords:
            row["gt_dtbi_y"], row["gt_dtbi_x"] = gt_coords["dtbi"]
        if "ptbi" in gt_coords:
            row["gt_ptbi_y"], row["gt_ptbi_x"] = gt_coords["ptbi"]

        if "left" in pred_coords:
            row["pred_dtbi_y"], row["pred_dtbi_x"] = pred_coords["left"]
        if "right" in pred_coords:
            row["pred_ptbi_y"], row["pred_ptbi_x"] = pred_coords["right"]

        if row["gt_dtbi_x"] is not None and row["pred_dtbi_x"] is not None:
            row["Distal Errors"] = np.sqrt(
                (row["gt_dtbi_y"] - row["pred_dtbi_y"]) ** 2 +
                (row["gt_dtbi_x"] - row["pred_dtbi_x"]) ** 2
            )
        if row["gt_ptbi_x"] is not None and row["pred_ptbi_x"] is not None:
            row["Proximal Errors"] = np.sqrt(
                (row["gt_ptbi_y"] - row["pred_ptbi_y"]) ** 2 +
                (row["gt_ptbi_x"] - row["pred_ptbi_x"]) ** 2
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    distal_mean = df["Distal Errors"].dropna().mean()
    proximal_mean = df["Proximal Errors"].dropna().mean()
    print(f"Distal mean error: {distal_mean:.2f} px")
    print(f"Proximal mean error: {proximal_mean:.2f} px")

    if save_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"CVS saved in {csv_path} with {len(df)} rows")
    else:
        print("Could not save CSV (save_csv = False)")

    return df

# Acceleration calculation (for setting a threshold on when to use Kalman filtered coords or the TransUNet predicted ones)
def calculate_acceleration_from_predictions(df, frame_rate=51, save_csv=False, csv_path="accelerations.csv"):
    delta_t = 1 / frame_rate

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
        dist_d = np.hypot(
            df.loc[i, 'pred_dtbi_x'] - df.loc[i-1, 'pred_dtbi_x'],
            df.loc[i, 'pred_dtbi_y'] - df.loc[i-1, 'pred_dtbi_y']
        )
        dist_p = np.hypot(
            df.loc[i, 'pred_ptbi_x'] - df.loc[i-1, 'pred_ptbi_x'],
            df.loc[i, 'pred_ptbi_y'] - df.loc[i-1, 'pred_ptbi_y']
        )

        distances_distal.append(dist_d)
        distances_proximal.append(dist_p)

        speed_d = dist_d / delta_t
        speed_p = dist_p / delta_t

        speed_distal.append(speed_d)
        speed_proximal.append(speed_p)

        accel_d = abs(speed_d - speed_distal[i-1]) / delta_t
        accel_p = abs(speed_p - speed_proximal[i-1]) / delta_t

        acceleration_distal.append(accel_d)
        acceleration_proximal.append(accel_p)

        dx_d = df.loc[i, 'pred_dtbi_x'] - df.loc[i-1, 'pred_dtbi_x']
        dy_d = df.loc[i, 'pred_dtbi_y'] - df.loc[i-1, 'pred_dtbi_y']
        dx_p = df.loc[i, 'pred_ptbi_x'] - df.loc[i-1, 'pred_ptbi_x']
        dy_p = df.loc[i, 'pred_ptbi_y'] - df.loc[i-1, 'pred_ptbi_y']

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

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Accelerations saved in: {csv_path}")

    return df

# Kalman filter application for the distal coords
def apply_kalman_filter_distal(df, cfg):
    umbral_acceleration = cfg.get("threshold_acceleration", 0)
    fps = cfg.get("frame_rate", 51)
    save_csv = cfg.get("save_csv", False)
    csv_path = cfg.get("output_csv_path", "kalman_distal.csv")
    save_plot = cfg.get("save_plot", False)
    plot_path = cfg.get("output_plot_path", "kalman_distal_plot.png")

    delta_t = 1 / fps

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([
        df['pred_dtbi_x'].iloc[0],     
        df['speed_distal_x'].iloc[0],  
        df['pred_dtbi_y'].iloc[0],     
        df['speed_distal_y'].iloc[0]   
    ])

    kf.F = np.array([
        [1, delta_t, 0,       0],
        [0, 1,       0,       0],
        [0, 0,       1, delta_t],
        [0, 0,       0,       1]
    ])

    kf.P = np.eye(4) * 0.1
    kf.Q = np.eye(4) * 0.1
    kf.R = np.eye(2) * 1.0
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    kalman_x = []
    kalman_y = []

    for _, row in df.iterrows():
        accel = row['distal_acceleration']
        kf.predict()

        if accel >= umbral_acceleration:
            z = np.array([row['pred_dtbi_x'], row['pred_dtbi_y']])
            kf.update(z)
            kalman_x.append(kf.x[0])
            kalman_y.append(kf.x[2])
        else:
            kalman_x.append(row['pred_dtbi_x'])
            kalman_y.append(row['pred_dtbi_y'])

    df['Kalman_Distal_X'] = kalman_x
    df['Kalman_Distal_Y'] = kalman_y

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Filtered coords saved in {csv_path}")

    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['pred_dtbi_x'], label='X original', color='blue')
        plt.plot(df.index, df['pred_dtbi_y'], label='Y original', color='green')
        plt.plot(df.index, kalman_x, label='X Kalman', color='red', linestyle='--')
        plt.plot(df.index, kalman_y, label='Y Kalman', color='orange', linestyle='--')
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("Position (px)")
        plt.title("Kalman filter - Distal")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved in: {plot_path}")

    return df

# Kalman filter application for the proximal coords
def apply_kalman_filter_proximal(df, cfg):
    umbral_acceleration = cfg.get("threshold_acceleration", 0)
    fps = cfg.get("frame_rate", 51)
    save_csv = cfg.get("save_csv", False)
    csv_path = cfg.get("output_csv_path", "kalman_proximal.csv")
    save_plot = cfg.get("save_plot", False)
    plot_path = cfg.get("output_plot_path", "kalman_proximal_plot.png")

    delta_t = 1 / fps

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([
        df['pred_ptbi_x'].iloc[0],       
        df['speed_proximal_x'].iloc[0], 
        df['pred_ptbi_y'].iloc[0],       
        df['speed_proximal_y'].iloc[0]  
    ])

    kf.F = np.array([
        [1, delta_t, 0,       0],
        [0, 1,       0,       0],
        [0, 0,       1, delta_t],
        [0, 0,       0,       1]
    ])

    kf.P = np.eye(4) * 0.1
    kf.Q = np.eye(4) * 0.1
    kf.R = np.eye(2) * 1.0
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    kalman_x = []
    kalman_y = []

    for _, row in df.iterrows():
        accel = row['proximal_acceleration']
        kf.predict()

        if accel >= umbral_acceleration:
            z = np.array([row['pred_ptbi_x'], row['pred_ptbi_y']])
            kf.update(z)
            kalman_x.append(kf.x[0])
            kalman_y.append(kf.x[2])
        else:
            kalman_x.append(row['pred_ptbi_x'])
            kalman_y.append(row['pred_ptbi_y'])

    df['Kalman_Proximal_X'] = kalman_x
    df['Kalman_Proximal_Y'] = kalman_y

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Filtered proximal coordinates saved in {csv_path}")

    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['pred_ptbi_x'], label='X original', color='blue')
        plt.plot(df.index, df['pred_ptbi_y'], label='Y original', color='green')
        plt.plot(df.index, kalman_x, label='X Kalman', color='red', linestyle='--')
        plt.plot(df.index, kalman_y, label='Y Kalman', color='orange', linestyle='--')
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("Position (px)")
        plt.title("Kalman filter - Proximal")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved in: {plot_path}")

    return df

# Video reconstruction
def reconstruct_video_with_overlay(test_videos_dir, ndjson_path, predictions_in_memory, video_name, output_video_file, frame_rate=51):
    with open(ndjson_path, "r", encoding="utf-8") as f:
        ndjson_data = [json.loads(line) for line in f]

    gt_by_video_frame = {}
    for entry in ndjson_data:
        vn = os.path.splitext(entry["data_row"]["external_id"])[0]
        project_id = next(iter(entry["projects"]))
        frames = entry["projects"][project_id]["labels"][0]["annotations"]["frames"]

        gt_by_video_frame[vn] = {}
        for frame_number, frame_data in frames.items():
            objs = frame_data.get("objects", {})
            coords = {}
            for obj in objs.values():
                coords[obj.get("value", "").lower()] = (
                    obj["point"]["y"],
                    obj["point"]["x"]
                )
            gt_by_video_frame[vn][int(frame_number) - 1] = coords

    preds_for_video = {}
    for pred_key, pred_data in predictions_in_memory.items():
        vn, frame_str = pred_key.rsplit("_frame_", 1)
        if vn != video_name:
            continue
        frame_idx = int(frame_str)
        preds_for_video[frame_idx] = pred_data.get("predicted_coords_scaled", {})

    video_path = os.path.join(test_videos_dir, video_name + ".mp4")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print(f"Could not open first frame of {video_name}")
        return

    height, width, _ = first_frame.shape
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw predictions (red)
        if frame_idx in preds_for_video:
            for _, (y_pred, x_pred) in preds_for_video[frame_idx].items():
                cv2.circle(frame, (int(x_pred), int(y_pred)), 5, (0, 0, 255), -1)

        # Draw labeled coords (green)
        if video_name in gt_by_video_frame and frame_idx in gt_by_video_frame[video_name]:
            for _, (y_gt, x_gt) in gt_by_video_frame[video_name][frame_idx].items():
                cv2.circle(frame, (int(x_gt), int(y_gt)), 5, (0, 255, 0), -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Reconstructed video saved in: {output_video_file}")
