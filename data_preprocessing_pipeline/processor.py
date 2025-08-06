import os
import cv2
import json
import numpy as np
import pandas as pd
import shutil

# Frames Preprocessing

def crop_video(input_video_path, output_video_path, json_output_path, final_width=508, final_height=632):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error when opening the video: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fixed_row = int(frame_height * 0.75)
    crop_bottom = 20

    def detect_cut_positions(frame, row, threshold=10):
        pixel_values = frame[row, :]
        cut_left = next((i for i in range(len(pixel_values)) if pixel_values[i] < threshold), None)
        cut_right = next((i for i in range(len(pixel_values) - 1, -1, -1) if pixel_values[i] < threshold), None)
        return cut_left, cut_right

    def detect_top_cut_position(frame, threshold=10):
        return next((row for row in range(frame.shape[0]) if np.any(frame[row, :] < threshold)), 0)

    processed_frames = []
    transform_data = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        cut_left, cut_right = detect_cut_positions(gray, fixed_row)
        cut_top = detect_top_cut_position(gray)
        if None in (cut_left, cut_right): continue

        cropped = frame[cut_top:-crop_bottom, cut_left:cut_right]
        ch, cw = cropped.shape[:2]
        scale = min(final_width / cw, final_height / ch)
        resized = cv2.resize(cropped, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

        pad_top = (final_height - resized.shape[0]) // 2
        pad_bottom = final_height - resized.shape[0] - pad_top
        pad_left = (final_width - resized.shape[1]) // 2
        pad_right = final_width - resized.shape[1] - pad_left

        final = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        processed_frames.append(final)

        transform_data[frame_idx] = {
            'cut_left': cut_left,
            'cut_top': cut_top,
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top
        }
        frame_idx += 1

    cap.release()
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_width, final_height))
    for f in processed_frames:
        out.write(f)
    out.release()

    with open(json_output_path, 'w') as f:
        json.dump(transform_data, f, indent=2)

    return processed_frames


def crop_all_videos(input_dir, output_dir, json_dir, final_width=508, final_height=632):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.lower().endswith(".mp4"):
            continue
        in_path = os.path.join(input_dir, file)
        name = os.path.splitext(file)[0]
        out_path = os.path.join(output_dir, f"{name}.mp4")
        json_path = os.path.join(json_dir, f"{name}_transform.json")
        print(f"Cropping: {file}")
        crop_video(in_path, out_path, json_path, final_width, final_height)

def extract_all_frames(input_dir, output_dir_base):
    os.makedirs(output_dir_base, exist_ok=True)

    for video_filename in os.listdir(input_dir):
        if not video_filename.lower().endswith('.mp4'):
            continue

        video_path = os.path.join(input_dir, video_filename)
        name = os.path.splitext(video_filename)[0]
        out_folder = os.path.join(output_dir_base, name)
        os.makedirs(out_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        print(f"Extracting frames from: {video_filename}")
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_path = os.path.join(out_folder, f"{name}_frame_{frame_number:04d}.png")
            cv2.imwrite(output_path, frame)
            frame_number += 1
        cap.release()
        print(f"{frame_number} frames saved in: {out_folder}")

def resize_all_frames_in_dir(input_root, output_root, target_height=512, target_width=512):
    os.makedirs(output_root, exist_ok=True)

    for video_folder in os.listdir(input_root):
        input_folder = os.path.join(input_root, video_folder)
        output_folder = os.path.join(output_root, video_folder)

        if not os.path.isdir(input_folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        print(f"🔄 Resizing frames from: {video_folder}")

        for filename in sorted(os.listdir(input_folder)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"⚠️ Could not load: {filename}")
                continue

            original_height, original_width = img.shape[:2]

            scale_factor = target_height / original_height
            new_width = int(round(original_width * scale_factor))

            resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)

            total_padding = target_width - new_width
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded_img = cv2.copyMakeBorder(
                resized_img,
                top=0,
                bottom=0,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, padded_img)

        print(f"✅ Finished: {output_folder}")

    print("🏁 Resizing complete.")

def resize_all_frames_in_dir(input_root, output_root, target_height=512, target_width=512):
    os.makedirs(output_root, exist_ok=True)

    for video_folder in os.listdir(input_root):
        input_folder = os.path.join(input_root, video_folder)
        output_folder = os.path.join(output_root, video_folder)

        if not os.path.isdir(input_folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        print(f"🔄 Resizing frames from: {video_folder}")

        for filename in sorted(os.listdir(input_folder)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)

            if img is None:
                print(f"⚠️ Could not load: {filename}")
                continue

            h_original, w_original = img.shape[:2]
            scale_factor = target_height / h_original
            new_width = int(round(w_original * scale_factor))

            resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)

            total_padding = target_width - new_width
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded_img = cv2.copyMakeBorder(
                resized_img,
                top=0,
                bottom=0,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, padded_img)

        print(f"✅ Finished: {output_folder}")

    print("🏁 Resize process complete.")

# Labels pre-processing

def convert_ndjson_to_json(ndjson_path, output_json_path):

    print(f"Converting NDJSON a JSON:\n📥 Input: {ndjson_path}\n Output: {output_json_path}")
    
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"JSON file saved as: {output_json_path}")

import json
import csv
import os

def json_to_coords_csv(json_path, output_csv_path):

    with open(json_path, 'r') as f:
        annotations = json.load(f)

    rows = []
    for entry in annotations:
        video_id = entry['data_row']['external_id'].replace('.mp4', '')

        project_id = next(iter(entry["projects"]))
        frames = entry["projects"][project_id]["labels"][0]["annotations"]["frames"]

        for frame_number, frame_data in frames.items():
            dtbi_coords = None
            ptbi_coords = None

            for obj in frame_data["objects"].values():
                if obj["value"] == "dtbi":
                    dtbi_coords = [obj["point"]["x"], obj["point"]["y"]]
                elif obj["value"] == "ptbi":
                    ptbi_coords = [obj["point"]["x"], obj["point"]["y"]]

            if dtbi_coords and ptbi_coords:
                rows.append({
                    "video_id": video_id,
                    "frame_id": int(frame_number),
                    "dtbi_x": dtbi_coords[0],
                    "dtbi_y": dtbi_coords[1],
                    "ptbi_x": ptbi_coords[0],
                    "ptbi_y": ptbi_coords[1]
                })

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "frame_id", "dtbi_x", "dtbi_y", "ptbi_x", "ptbi_y"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"csv with coordinates saved in: {output_csv_path}")


def scale_coords_csv(input_csv_path, output_csv_path, orig_size=(632, 508), target_size=(512, 512)):
    import pandas as pd
    import os

    orig_h, orig_w = orig_size
    target_h, target_w = target_size

    scale_y = target_h / orig_h
    scaled_w = orig_w * scale_y
    pad_x = (target_w - scaled_w) / 2

    df = pd.read_csv(input_csv_path)

    df['dtbi_x'] = df['dtbi_x'] * scale_y + pad_x
    df['dtbi_y'] = df['dtbi_y'] * scale_y
    df['ptbi_x'] = df['ptbi_x'] * scale_y + pad_x
    df['ptbi_y'] = df['ptbi_y'] * scale_y

    #df = df.round({'dtbi_x': 2, 'dtbi_y': 2, 'ptbi_x': 2, 'ptbi_y': 2})

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"New scaled coords saved in {output_csv_path}")


def generate_masks(csv_path, output_base_folder, frames_base_dir=None, target_size=512, radio=30):
    import os
    import csv
    import cv2
    import numpy as np

    os.makedirs(output_base_folder, exist_ok=True)
    warned_videos = set()  

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            frame_id = int(row['frame_id']) - 1  

            if frames_base_dir:
                frame_filename = f"{video_id}_frame_{frame_id:04d}.png"
                frame_path = os.path.join(frames_base_dir, video_id, frame_filename)
                if not os.path.exists(frame_path):
                    continue

            video_folder = os.path.join(output_base_folder, video_id)
            os.makedirs(video_folder, exist_ok=True)

            mask_filename = f"{video_id}_frame_{frame_id:04d}_mask.png"
            mask_path = os.path.join(video_folder, mask_filename)

            mask = np.zeros((target_size, target_size), dtype=np.uint8)

            for punto in ['dtbi', 'ptbi']:
                x_key = f"{punto}_x"
                y_key = f"{punto}_y"
                if row[x_key] and row[y_key]:
                    x = int(round(float(row[x_key])))
                    y = int(round(float(row[y_key])))
                    cv2.circle(mask, (x, y), radio, 255, -1)

            cv2.imwrite(mask_path, mask)

    print(f"Masks generated and saved in {output_base_folder}")


def generate_npz(csv_path, frames_folder, masks_folder,
                 output_train_folder, output_test_folder,
                 num_test_videos=2):
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Filtrar solo los videos realmente presentes en frames_folder
    available_videos = [v for v in df['video_id'].unique()
                        if os.path.exists(os.path.join(frames_folder, v))]
    np.random.shuffle(available_videos)

    test_ids = set(available_videos[:num_test_videos])
    train_ids = set(available_videos[num_test_videos:])

    for video_id in available_videos:
        subset = df[df['video_id'] == video_id]
        output_base = output_test_folder if video_id in test_ids else output_train_folder

        for _, row in subset.iterrows():
            frame_id = int(row['frame_id']) - 1
            frame_name = f"{video_id}_frame_{frame_id:04d}.png"
            mask_name = f"{video_id}_frame_{frame_id:04d}_mask.png"

            frame_path = os.path.join(frames_folder, video_id, frame_name)
            mask_path = os.path.join(masks_folder, video_id, mask_name)

            if not os.path.exists(frame_path) or not os.path.exists(mask_path):
                print(f"⚠️ Files not found for frame {frame_name}")
                continue

            image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or label is None:
                print(f"⚠️ Error loading frame or mask {frame_name}")
                continue

            dtbi = (float(row['dtbi_x']), float(row['dtbi_y']))
            ptbi = (float(row['ptbi_x']), float(row['ptbi_y']))
            insertion_coords = np.array([dtbi, ptbi], dtype=np.float32)

            npz_path = os.path.join(output_base, f"{video_id}_frame_{frame_id:04d}.npz")
            np.savez_compressed(npz_path,
                                image=image,
                                label=label,
                                insertion_coords=insertion_coords)

    print(f"NPZ dataset generation finished.\nTrain folder: {output_train_folder}\nTest folder: {output_test_folder}")


def generate_npz_lists(train_npz_folder, test_npz_folder,
                       train_list_output_folder, test_list_output_folder,
                       train_txt_output='train.txt', test_txt_output='test_vol.txt'):

    def write_list(source_folder, output_folder, output_file):
        os.makedirs(output_folder, exist_ok=True)
        filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(source_folder)
            if f.endswith('.npz')
        ]
        output_path = os.path.join(output_folder, output_file)
        with open(output_path, 'w') as f:
            f.write('\n'.join(sorted(filenames)))
        print(f"📄 {output_file} generado con {len(filenames)} elementos en {output_folder}")

    write_list(train_npz_folder, train_list_output_folder, train_txt_output)
    write_list(test_npz_folder, test_list_output_folder, test_txt_output)


def clean_temp_folder(temp_folder_path):
    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
        print(f"Temp folder eliminated: {temp_folder_path}")
    else:
        print(f"Temp folder nor found: {temp_folder_path}")