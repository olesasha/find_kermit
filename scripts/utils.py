import re
import os 
import cv2
from tqdm import tqdm
import shutil
import random
import pandas as pd

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
        [f for f in os.listdir(frames_path) if f.endswith(".png")],
        key=lambda x: (
            x.split("_")[0],  # Sort by video identifier (e.g., "Muppets-02-01-01")
            int(x.split("_frame")[1].split(".png")[0])  # Then by frame number
        )
    )
    
    total_frames = len(frame_files)
    if total_frames == 0:
        raise ValueError("No frames found in the specified directory.")
    
    # calculate number of frames for validation and test sets
    num_val_frames = int(total_frames * val_size)
    num_test_frames = int(total_frames * test_size)
    
    if num_val_frames + num_test_frames >= total_frames:
        raise ValueError("Validation and test percentages must sum to less than 1.")
    
    # shuffle the frame files for randomness
    random.Random(42).shuffle(frame_files)
    
    # split frames into train, validation, and test sets
    test_frames = frame_files[:num_test_frames]
    val_frames = frame_files[num_test_frames:num_test_frames + num_val_frames]
    train_frames = frame_files[num_test_frames + num_val_frames:]
    
    # move frames to respective directories
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
    pattern = re.compile(r'.*\.png$')
    for root, _, files in os.walk(path):
        for file in files:
            if pattern.match(file): 
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
                    