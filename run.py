import yaml
import os
import glob
from datetime import datetime
import shutil
from data_preprocessing_pipeline.processor import (
    crop_all_videos, extract_all_frames, resize_all_frames_in_dir,
    convert_ndjson_to_json, json_to_coords_csv, scale_coords_csv,
    generate_masks, generate_npz, generate_npz_lists, clean_temp_folder
)
from ml_module.train import run_training_pipeline
from ml_module.test import run_testing_pipeline
from post_processing_pipeline.post_processor import calculate_prediction_errors, reconstruct_video_from_frames, run_postprocessing_pipeline, calculate_acceleration_from_predictions, apply_kalman_filter_distal, apply_kalman_filter_proximal, reconstruct_video_from_frames, clean_pipeline_outputs


def run_pipeline(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    print(f"Config saved in {run_dir}/config.yaml")

    preprocessing = config.get('preprocessing', {})
    training = config.get('training', {})
    postprocessing = config.get("post_processing", {})


    # --- PREPROCESSING ---
    if preprocessing.get("enabled", False):

        if preprocessing.get('video_cropping_batch', {}).get('enabled', False):
            cfg = preprocessing['video_cropping_batch']
            crop_all_videos(
                input_dir=cfg['input_dir'],
                output_dir=cfg['output_dir'],
                json_dir=cfg['json_dir'],
                final_width=cfg['resolution'][0],
                final_height=cfg['resolution'][1]
            )

        if preprocessing.get('extract_frames_batch', {}).get('enabled', False):
            cfg = preprocessing['extract_frames_batch']
            extract_all_frames(
                input_dir=cfg['input_dir'],
                output_dir_base=cfg['output_dir']
            )

        if preprocessing.get('resize_frames', {}).get('enabled', False):
            cfg = preprocessing['resize_frames']
            resize_all_frames_in_dir(
                input_root=cfg['input_dir'],
                output_root=cfg['output_dir'],
                target_height=cfg['target_size'][0],
                target_width=cfg['target_size'][1]
            )

        if preprocessing.get('labelbox_conversion', {}).get('enabled', False):
            cfg = preprocessing['labelbox_conversion']
            ndjson_dir = cfg['ndjson_path']
            output_json_path = cfg['output_json_path']
            ndjson_files = glob.glob(os.path.join(ndjson_dir, "*.ndjson"))
            if not ndjson_files:
                raise FileNotFoundError(f"No NDJSON file found in {ndjson_dir}")
            ndjson_path = ndjson_files[0]
            print(f"📂 Archivo NDJSON detectado: {ndjson_path}")
            convert_ndjson_to_json(ndjson_path, output_json_path)

        if preprocessing.get("coordinate_extraction", {}).get("enabled", False):
            cfg = preprocessing["coordinate_extraction"]
            json_to_coords_csv(cfg["input_json"], cfg["output_csv"])

        if preprocessing.get("scale_coordinates", {}).get("enabled", False):
            cfg = preprocessing["scale_coordinates"]
            scale_coords_csv(
                input_csv_path=cfg['input_csv'],
                output_csv_path=cfg['output_csv'],
                orig_size=tuple(cfg.get('original_size', [632, 508])),
                target_size=tuple(cfg.get('target_size', [512, 512]))
            )

        if preprocessing.get("generate_masks", {}).get("enabled", False):
            cfg = preprocessing["generate_masks"]
            generate_masks(
                csv_path=cfg['input_csv'],
                output_base_folder=cfg['output_dir'],
                frames_base_dir=cfg.get('frames_dir', None),
                target_size=cfg.get('target_size', 512),
                radio=cfg.get('radius', 30)
            )

        if preprocessing.get("generate_npz", {}).get("enabled", False):
            cfg = preprocessing["generate_npz"]
            generate_npz(
                csv_path=cfg['input_csv'],
                frames_folder=cfg['frames_folder'],
                masks_folder=cfg['masks_folder'],
                output_train_folder=cfg['output_train_folder'],
                output_test_folder=cfg['output_test_folder'],
                num_test_videos=cfg.get('num_test_videos', 2)
            )

        if preprocessing.get("generate_npz_list", {}).get("enabled", False):
            cfg = preprocessing["generate_npz_list"]
            generate_npz_lists(
                train_npz_folder=cfg["train_npz_folder"],
                test_npz_folder=cfg["test_npz_folder"],
                train_list_output_folder=cfg["train_list_output_folder"],
                test_list_output_folder=cfg["test_list_output_folder"],
                train_txt_output=cfg.get("train_txt_output", "train.txt"),
                test_txt_output=cfg.get("test_txt_output", "test_vol.txt")
            )

        if preprocessing.get("cleanup", {}).get("enabled", False):
            cfg = preprocessing["cleanup"]
            clean_temp_folder(temp_folder_path=cfg["temp_folder"])

    # --- TRAINING ---
    if training.get("enabled", False):
        run_training_pipeline(training)

    # --- TESTING ---
    if config.get("testing", {}).get("enabled", False):
        run_testing_pipeline(config["testing"])

    # --- POST-PROCESSING ---
    if postprocessing.get("enabled", False):

        if postprocessing.get("cleanup_pipeline", {}).get("enabled", False):
            print("🧼 Limpiando archivos y carpetas del pipeline...")
            clean_pipeline_outputs(postprocessing["cleanup_pipeline"])
            
        if postprocessing.get("find_center_of_mass", {}).get("enabled", False):
            print("Finding centers of mass for hottest pixels...")
            cfg = postprocessing["find_center_of_mass"]
            run_postprocessing_pipeline(cfg)

        if postprocessing.get("calculate_prediction_errors", {}).get("enabled", False):
            print("Calculating prediction errors...")
            calculate_prediction_errors(postprocessing["calculate_prediction_errors"])

        if postprocessing.get("calculate_acceleration", {}).get("enabled", False):
            print("Calculating accelerations...")
            calculate_acceleration_from_predictions(postprocessing["calculate_acceleration"])

        if postprocessing.get("kalman_filter_distal", {}).get("enabled", False):
            print("Applying Kalman filter to distal coordinates...")
            apply_kalman_filter_distal(postprocessing["kalman_filter_distal"])

        if postprocessing.get("kalman_filter_proximal", {}).get("enabled", False):
            print("Applying Kalman filter to proximal coordinates...")
            apply_kalman_filter_proximal(postprocessing["kalman_filter_proximal"])

        if postprocessing.get("reconstruct_video", {}).get("enabled", False):
            print("Reconstructing video from frames...")
            reconstruct_video_from_frames(postprocessing["reconstruct_video"])



if __name__ == "__main__":
    run_pipeline()
