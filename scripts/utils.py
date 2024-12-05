import re
import os 
import cv2
from tqdm import tqdm

def extract_frames(data_path: str, start_time: int = 15, frame_interval: float = 0.5):
    """
    Extract frames from a video, sampling every `frame_interval` seconds,
    and skipping the first `start_time` seconds.
    
    Parameters:
    - data_path (str): Path to the video file.
    - start_time (int): Time in seconds to skip from the start of the video. Default: 15 seconds.
    - frame_interval (float): Time interval in seconds to sample frames. Default: half a second.
    """
    capture = cv2.VideoCapture(data_path)
    fps = capture.get(cv2.CAP_PROP_FPS)  
    if not capture.isOpened():
        print("Error: video file not accessible.")
        return

    # define starting point
    capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000) # convert to ms
    
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # extract samples every frame_interval
    total_samples = int((duration - start_time) / frame_interval)
    
    # logging with tqdm
    pbar = tqdm(total=total_samples, desc="Extracting frames")
    frame_nr = 0
    current_time = start_time * 1000  

    # iterate through video and extract frames
    while current_time < duration * 1000:
        capture.set(cv2.CAP_PROP_POS_MSEC, current_time)
        success, frame = capture.read()

        if success:
            # save frame in target repo
            cv2.imwrite(f'../ground_truth_data/frame{frame_nr}.png', frame)
            frame_nr += 1

        current_time += frame_interval * 1000 # convert to ms
        pbar.update(1)

    capture.release()
    pbar.close()
    print(f"Frames extracted successfully, sampled every {frame_interval} seconds, skipping the first {start_time} seconds.")
    

import os
import shutil
import random

def train_test_split(frames_path: str, train_path: str, val_path: str, test_path: str, val_size: float, test_size: float):
    """
    Splits the frames in the repository into train, validation, and test splits.
    Moves the frames from the origin directory to the respective target directories.
    
    Parameters:
    - frames_path (str): Path to the directory containing the frames.
    - train_path (str): Path to the directory for the training set.
    - val_path (str): Path to the directory for the validation set.
    - test_path (str): Path to the directory for the test set.
    - val_size (float): Fraction of frames to allocate for validation (0 < val_size < 1).
    - test_size (float): Fraction of frames to allocate for testing (0 < test_size < 1).
    """
    
    frame_files = sorted(
        [f for f in os.listdir(frames_path) if f.startswith("frame") and f.endswith(".png")],
        key=lambda x: int(x[5:-4]) 
    )
    
    total_frames = len(frame_files)
    if total_frames == 0:
        raise ValueError("No frames found in the specified directory.")
    
    num_val_frames = int(total_frames * val_size)
    num_test_frames = int(total_frames * test_size)
    
    if num_val_frames + num_test_frames >= total_frames:
        raise ValueError("Validation and test percentages must sum to less than 1.")
    
    random.Random(42).shuffle(frame_files)
    
    test_frames = frame_files[:num_test_frames]
    val_frames = frame_files[num_test_frames:num_test_frames + num_val_frames]
    train_frames = frame_files[num_test_frames + num_val_frames:]
    
    for frame in train_frames:
        shutil.move(os.path.join(frames_path, frame), os.path.join(train_path, frame))
    for frame in val_frames:
        shutil.move(os.path.join(frames_path, frame), os.path.join(val_path, frame))
    for frame in test_frames:
        shutil.move(os.path.join(frames_path, frame), os.path.join(test_path, frame))
    
    print(f"Split completed: {len(train_frames)} train, {len(val_frames)} validation, {len(test_frames)} test frames.")

    
def delete_frames(path:str):
    """
    Removes generated frames from the repository of choice.
    - path (str): path to the directory with frames that are to be removed
    """
    pattern = re.compile(r'frame\d+\.png')

    for root, _, files in os.walk(path):
        for file in files:
            if pattern.match(file): 
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
                    