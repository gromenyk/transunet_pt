import cv2
import os
import numpy as np
import argparse

video_path = './datasets/videos/example_video.mp4'
output_folder = './data/Synapse/test_vol_h5'
original_images_file = '../data/Synapse/original_images.npy'

cap = cv2.VideoCapture(video_path)

def frame_split(video_path, output_folder, original_images_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.isfile(video_path):
        print(f'Error: Video file {video_path} does not exist')
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Could not find any videos')
        return

    frame_count = 0
    original_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #Resize frame keeping proportion
        height, width, channels = frame.shape
        max_side = max(height, width)
        scale = 512 / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        #Padding addition
        top = (512 - new_height) // 2
        bottom = 512 - new_height - top
        left = (512 - new_width) // 2
        right = 512 - new_width - left
        padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

        # Save original images
        original_images.append(padded_frame)

        #Image normalization (zscore)
        mean = np.mean(padded_frame)
        std = np.std(padded_frame)
        normalized_image = (padded_frame - mean) / std

        #Create normalized empty mask
        empty_mask = np.zeros((512,512), dtype=np.uint8)

        npz_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.npz')
        np.savez(npz_filename, image=normalized_image, label=empty_mask, mean=mean, std=std)

        frame_count += 1
    
    cap.release()

    np.save(original_images_file, np.array(original_images))

    print(f'Extraction Completed. {frame_count} frames and masks were generated')
