import os
import cv2
import json
import numpy as np
import pandas as pd
import shutil

# If it is a video just fresh out of the device, we can crop it in memory

def crop_video_in_memory(input_video_path, final_width=508, final_height=632):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Error opening video: {input_video_path}")

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
        if None in (cut_left, cut_right):
            continue

        cropped = frame[cut_top:-crop_bottom, cut_left:cut_right]
        ch, cw = cropped.shape[:2]
        scale = min(final_width / cw, final_height / ch)
        resized = cv2.resize(cropped, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

        pad_top = (final_height - resized.shape[0]) // 2
        pad_bottom = final_height - resized.shape[0] - pad_top
        pad_left = (final_width - resized.shape[1]) // 2
        pad_right = final_width - resized.shape[1] - pad_left

        final = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=0)
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

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    return {
        "video_name": video_name,
        "frames": processed_frames,
        "fps": fps,
        "transform_data": transform_data
    }

def crop_all_videos_in_memory(input_dir, final_width=508, final_height=632, save_debug=False, debug_dir="debug"):
    processed_videos = {}

    for idx, file in enumerate(sorted(os.listdir(input_dir))):
        if not file.lower().endswith(".mp4"):
            continue

        in_path = os.path.join(input_dir, file)
        print(f"📼 Cropping in memory: {file}")
        result = crop_video_in_memory(in_path, final_width, final_height)

        video_name = result['video_name']
        processed_videos[video_name] = result

        if save_debug and idx == 0 and result['frames']:
            os.makedirs(debug_dir, exist_ok=True)
            sample_frame = result['frames'][0]
            cv2.imwrite(os.path.join(debug_dir, f"{video_name}_frame_0000.png"), sample_frame)
            with open(os.path.join(debug_dir, f"{video_name}_transform.json"), 'w') as f:
                json.dump(result['transform_data'], f, indent=2)

    return processed_videos  # Dictionary with data per video.


# Extract frames from a video and resize them in memory

def extract_frames_in_memory(input_dir, save_debug=False, debug_dir="debug"):
    extracted = {}

    for idx, file in enumerate(sorted(os.listdir(input_dir))):
        if not file.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(input_dir, file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening: {file}")
            continue

        video_name = os.path.splitext(file)[0]
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_idx += 1

        cap.release()
        extracted[video_name] = frames

        if save_debug and frames:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{video_name}_frame_0000.png"), frames[0])

        print(f"Extracted {len(frames)} frames from {file}")

    return extracted


# Resize frames in memory to a target size, padding if necessary

def resize_frames_in_memory(frames, target_height=512, target_width=512, save_debug=False, debug_dir="debug", video_name=""):
    resized_frames = []

    for idx, img in enumerate(frames):
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

        resized_frames.append(padded_img)

        if save_debug and idx == 0:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            out_path = f"{debug_dir}/{video_name}_resized_frame_0000.png"
            cv2.imwrite(out_path, padded_img)

    return resized_frames

# Load NDJSON annotations

def load_ndjson(ndjson_path):
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

# Extract coordinates from NDJSON and save them in a dictionary

def extract_coords_from_ndjson(data):
    coords_by_video = {}

    for entry in data:
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
                coords_by_video.setdefault(video_id, []).append({
                    "frame_id": int(frame_number),
                    "dtbi_x": dtbi_coords[0],
                    "dtbi_y": dtbi_coords[1],
                    "ptbi_x": ptbi_coords[0],
                    "ptbi_y": ptbi_coords[1]
                })

    return coords_by_video

# Transform coordinates to match the resized frames

def scale_coords_dict_in_memory(coords_by_video, orig_size=(632, 508), target_size=(512, 512)):
    orig_h, orig_w = orig_size
    target_h, target_w = target_size

    scale_y = target_h / orig_h
    scaled_w = orig_w * scale_y
    pad_x = (target_w - scaled_w) / 2

    scaled_coords_by_video = {}

    for video_id, frames in coords_by_video.items():
        scaled_coords_by_video[video_id] = []

        for row in frames:
            scaled_row = {
                "frame_id": row["frame_id"],
                "dtbi_x": row["dtbi_x"] * scale_y + pad_x,
                "dtbi_y": row["dtbi_y"] * scale_y,
                "ptbi_x": row["ptbi_x"] * scale_y + pad_x,
                "ptbi_y": row["ptbi_y"] * scale_y,
            }
            scaled_coords_by_video[video_id].append(scaled_row)

    return scaled_coords_by_video

# Masks generation from coordinates

def generate_masks_in_memory(scaled_coords_by_video, target_size=512, radius=30, save_debug=False, debug_dir=None):
    masks_by_video = {}

    for video_id, coords_list in scaled_coords_by_video.items():
        num_frames = max(c["frame_id"] for c in coords_list) + 1
        masks = [np.zeros((target_size, target_size), dtype=np.uint8) for _ in range(num_frames)]

        for coord in coords_list:
            frame_id = coord["frame_id"] - 1
            for punto in ["dtbi", "ptbi"]:
                x = int(round(coord[f"{punto}_x"]))
                y = int(round(coord[f"{punto}_y"]))
                if 0 <= x < target_size and 0 <= y < target_size:
                    cv2.circle(masks[frame_id], (x, y), radius, 255, -1)

        masks_by_video[video_id] = masks

        if save_debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"{video_id}_mask_frame_0000.png")
            cv2.imwrite(debug_path, masks[0])
            print(f"Debug: first mask saved in {debug_path}")

    return masks_by_video

# Generate NPZ files.

def generate_npz_in_memory(processed_videos, masks_by_video, scaled_coords_by_video,
                           mode, save=False, output_folder=None):

    npz_items = []

    for video_id in processed_videos.keys():
        frames = processed_videos[video_id]["frames"]
        masks = masks_by_video.get(video_id)
        coords = scaled_coords_by_video.get(video_id)

        if masks is None or coords is None:
            print(f"Skipping {video_id} — missing masks or coords")
            continue

        for row in coords:
            frame_id = row["frame_id"]
            frame_idx = frame_id - 1

            if frame_idx < 0 or frame_idx >= len(frames) or frame_idx >= len(masks):
                print(f"Frame {frame_id} (adjusted to {frame_idx}) out of bounds in {video_id}")
                continue

            image = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2GRAY) if len(frames[frame_idx].shape) == 3 else frames[frame_idx]
            label = masks[frame_idx]
            insertion_coords = np.array([
                (row["dtbi_x"], row["dtbi_y"]),
                (row["ptbi_x"], row["ptbi_y"])
            ], dtype=np.float32)

            item = {
                "video_id": video_id,
                "frame_id": frame_id,
                "image": image,
                "label": label,
                "insertion_coords": insertion_coords
            }
            npz_items.append(item)

            if save:
                os.makedirs(output_folder, exist_ok=True)
                filename = f"{video_id}_frame_{frame_idx:04d}.npz"
                np.savez_compressed(os.path.join(output_folder, filename),
                                    image=image,
                                    label=label,
                                    insertion_coords=insertion_coords)

    print(f"NPZ Dataset ready to go. {mode.capitalize()}: {len(npz_items)}")
    return npz_items

# Generate NPZ files list

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
        print(f"{output_file} generated with {len(filenames)} elements in {output_folder}")

    write_list(train_npz_folder, train_list_output_folder, train_txt_output)
    write_list(test_npz_folder, test_list_output_folder, test_txt_output)