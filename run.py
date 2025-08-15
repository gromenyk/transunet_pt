import yaml
from datetime import datetime
import shutil
import os
import cv2
from data_preprocessing_pipeline.preprocessor_memory import *
from ml_module.train import run_training_pipeline
from ml_module.test import run_testing_pipeline
from post_processing_pipeline.post_processor_memory import *

print(">>> THIS IS THE CORRECT RUN.PY <<<")
print("File path:", os.path.abspath(__file__))

def save_run_config(config_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    run_dir = os.path.join(runs_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    print(f"Configuration saved in {run_dir}/config.yaml")
    return run_dir

def run_pipeline(config_path="config.yaml"):
    run_dir = save_run_config(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Loaded config from:", os.path.abspath(config_path))
    print("pipeline.enabled in loaded config:", config.get("pipeline", {}).get("enabled"))


    pipeline = config.get("pipeline", {})
    training_cfg = config.get("training", {})
    testing_cfg = config.get("testing", {})
    post_cfg = config.get("post_processing", {})

    debug = config.get("debug", {})
    debug_enabled = debug.get("enabled", False)
    debug_dir = debug.get("dir", "debug/")

    # --- Preprocessing ---
    if pipeline.get("enabled", False):
        def process_set(input_dir, mode, output_npz_dir):
            print(f"\n=== Processing {mode.upper()} set from {input_dir} ===")
            processed_videos = {}

            # Step 1: Crop or load
            if pipeline.get("crop_videos", {}).get("enabled", False):
                crop_cfg = pipeline["crop_videos"]
                processed_videos = crop_all_videos_in_memory(
                    input_dir=input_dir,
                    final_width=crop_cfg["final_width"],
                    final_height=crop_cfg["final_height"],
                    save_debug=debug_enabled,
                    debug_dir=os.path.join(debug_dir, f"frames_{mode}")
                )
            else:
                print(f"Skipping cropping for {mode}, loading videos directly.")
                for file in sorted(os.listdir(input_dir)):
                    if not file.lower().endswith(".mp4"):
                        continue
                    video_path = os.path.join(input_dir, file)
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Cannot open {file}")
                        continue
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    video_name = os.path.splitext(file)[0]
                    processed_videos[video_name] = {
                        "video_name": video_name,
                        "frames": frames,
                        "fps": fps,
                        "transform_data": None
                    }

            # Step 2: Resize
            if pipeline.get("resize_frames", {}).get("enabled", False):
                resize_cfg = pipeline["resize_frames"]
                for video_id in processed_videos:
                    resized = resize_frames_in_memory(
                        processed_videos[video_id]["frames"],
                        target_height=resize_cfg["target_height"],
                        target_width=resize_cfg["target_width"],
                        save_debug=debug_enabled,
                        debug_dir=os.path.join(debug_dir, f"resized_{mode}"),
                        video_name=video_id
                    )
                    processed_videos[video_id]["frames"] = resized

            # Step 3: Load NDJSON
            ndjson_data = None
            if pipeline.get("load_ndjson", {}).get("enabled", False):
                ndjson_data = load_ndjson(pipeline["load_ndjson"]["path"])

            # Step 4: Extract coords
            coords_by_video = {}
            if pipeline.get("extract_coords", {}).get("enabled", False):
                coords_by_video = extract_coords_from_ndjson(ndjson_data)

            # Step 5: Scale coords
            scaled_coords_by_video = {}
            if pipeline.get("scale_coords", {}).get("enabled", False):
                scale_cfg = pipeline["scale_coords"]
                scaled_coords_by_video = scale_coords_dict_in_memory(
                    coords_by_video,
                    orig_size=tuple(scale_cfg["input_size"]),
                    target_size=tuple(scale_cfg["target_size"])
                )

            # Step 6: Generate masks
            masks_by_video = {}
            if pipeline.get("generate_masks", {}).get("enabled", False):
                masks_by_video = generate_masks_in_memory(
                    scaled_coords_by_video,
                    target_size=pipeline["scale_coords"]["target_size"][0],
                    radius=pipeline["generate_masks"]["radius"],
                    save_debug=debug_enabled,
                    debug_dir=os.path.join(debug_dir, f"masks_{mode}")
                )

            # Step 7: NPZ generation
            if pipeline.get("generate_npz", {}).get("enabled", False):
                generate_npz_in_memory(
                    processed_videos,
                    masks_by_video,
                    scaled_coords_by_video,
                    mode=mode,
                    save=True,
                    output_folder=output_npz_dir
                )

        # Process train and test
        npz_cfg = pipeline.get("generate_npz", {})
        process_set("data_preprocessing_pipeline/raw_data/train_videos", "train", npz_cfg.get("output_train_dir", ""))
        process_set("data_preprocessing_pipeline/raw_data/test_videos", "test", npz_cfg.get("output_test_dir", ""))

        # Step 8: NPZ lists
        if pipeline.get("generate_npz_lists", {}).get("enabled", False):
            lists_cfg = pipeline["generate_npz_lists"]
            generate_npz_lists(
                train_npz_folder=lists_cfg["train_npz_folder"],
                test_npz_folder=lists_cfg["test_npz_folder"],
                train_list_output_folder=lists_cfg["train_list_output_folder"],
                test_list_output_folder=lists_cfg["test_list_output_folder"],
                train_txt_output=lists_cfg.get("train_txt_output", "train.txt"),
                test_txt_output=lists_cfg.get("test_txt_output", "test_vol.txt")
            )

    else:
        print("⚠️ Skipping pre-processing pipeline (pipeline.enabled = false)")

    # --- Training ---
    if training_cfg.get("enabled", False):
        print("🚀 Starting training pipeline...")
        run_training_pipeline(training_cfg)

    # --- Testing ---
    predictions_in_memory = {}
    if testing_cfg.get("enabled", False):
        print("🚀 Starting testing pipeline...")
        predictions_in_memory = run_testing_pipeline(testing_cfg)
        print("Type of predictions_in_memory:", type(predictions_in_memory))
        if isinstance(predictions_in_memory, dict):
            print("Elements quantiy:", len(predictions_in_memory))
            first_key = next(iter(predictions_in_memory))
            print("First key:", first_key)
            first_value = predictions_in_memory[first_key]
            print("Type of the fisrt value:", type(first_value))
            if isinstance(first_value, dict):
                print("Internal keys of the first value:", first_value.keys())
            else:
                print("Content of the first value:", first_value)
        else:
            print("Not a dictionary")

    # --- Post-processing ---
    # Step 1: Find centers of mass (predictions 224x224)
    for key, data in predictions_in_memory.items():
        prediction_array = data.get("prediction")
        if prediction_array is None:
            continue

        left_center, right_center = find_centers_of_mass_for_hottest_pixels(prediction_array)
        data["predicted_coords"] = {
            "left": left_center,
            "right": right_center
        }

    # Step 2: Scale predicted coords to 632x508
    predictions_in_memory = scale_predicted_coords(predictions_in_memory, target_size=(632, 508))

    # Step 2.1: Visual debug
    if post_cfg.get("debug_overlay", {}).get("enabled", True):
        draw_gt_and_predictions(
            test_videos_dir=post_cfg["debug_overlay"]["test_videos_dir"],
            ndjson_path=post_cfg["debug_overlay"]["ndjson_path"],
            predictions_in_memory=predictions_in_memory,
            output_dir=post_cfg["debug_overlay"]["output_dir"]
        )

    # Step 3: Calculate errors
    if post_cfg.get("calculate_errors", {}).get("enabled", True):
        gt_coords_by_video = load_gt_coords(
            ndjson_path=post_cfg["calculate_errors"]["ndjson_path"]
        )
        df = calculate_errors(
            predictions_in_memory,
            gt_coords_by_video,
            save_csv=post_cfg["calculate_errors"].get("save_csv", True),
            csv_path=post_cfg["calculate_errors"].get("csv_path", "errors.csv")
        )

    # Step 4: Calculate accelerations
    if post_cfg.get("calculate_accelerations", {}).get("enabled", False):
        df = calculate_acceleration_from_predictions(
            df,
            frame_rate=post_cfg["calculate_accelerations"].get("frame_rate", 51),
            save_csv=post_cfg["calculate_accelerations"].get("save_csv", True),
            csv_path=post_cfg["calculate_accelerations"].get("csv_path", "accelerations.csv")
        )
        print(f"✅ Accelerations calculated. DataFrame shape: {df.shape}")

    # Step 5: Apply Kalman filter to distal predicted coords.
    if post_cfg.get("kalman_filter_distal", {}).get("enabled", False):
        df = apply_kalman_filter_distal(df, post_cfg["kalman_filter_distal"])

    # Step 6: Apply Kalma filter to proximal predicted coords.
    if post_cfg.get("kalman_filter_proximal", {}).get("enabled", False):
        df = apply_kalman_filter_proximal(df, post_cfg["kalman_filter_proximal"])

    # Reconstruct video with TransUNet predicted coords.
    if post_cfg.get("reconstruct_video", {}).get("enabled", False):
        print("Reconstructing video...")

        test_videos_dir = post_cfg["debug_overlay"]["test_videos_dir"]
        ndjson_path = post_cfg["debug_overlay"]["ndjson_path"]
        output_dir = post_cfg["reconstruct_video"]["output_dir"]
        frame_rate = post_cfg["reconstruct_video"].get("frame_rate", 51)

        os.makedirs(output_dir, exist_ok=True)

        video_names = sorted({key.rsplit("_frame_", 1)[0] for key in predictions_in_memory.keys()})

        for video_name in video_names:
            output_path = os.path.join(output_dir, f"{video_name}_overlay.mp4")
            reconstruct_video_with_overlay(
                test_videos_dir=test_videos_dir,
                ndjson_path=ndjson_path,
                predictions_in_memory=predictions_in_memory,
                video_name=video_name,
                output_video_file=output_path,
                frame_rate=frame_rate
            )

    
if __name__ == "__main__":
    run_pipeline(config_path='config.yaml')
