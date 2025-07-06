import re
import os 
import cv2
from tqdm import tqdm
import shutil
import random
import pandas as pd

def extract_frames(muppet_files: dict, data_path: str = "../ground_truth_data", output_dir: str = "../ground_truth_data/frames"):
    """
    Extract frames from videos specified in `muppet_files`, aligning each frame with rows in the 
    corresponding annotation file. 
    
    Parameters:
    - muppet_files (dict): Dictionary mapping video file names to annotation file paths.
    - data_path (str): Directory containing the video and annotation files.
    - output_dir (str): Directory to save the extracted frames. 
    """
    
    for video_file, annotation_file in muppet_files.items():

        video_path = os.path.join(data_path, video_file)
        annotation_path = os.path.join(data_path, annotation_file)
        
        # checks for the existence of videos and annotations
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            continue
        
        if not os.path.exists(annotation_path):
            print(f"Error: Annotation file '{annotation_path}' not found.")
            continue
        
        # load annotations
        annotations = pd.read_csv(annotation_path, sep=";")
        total_frames = annotations.shape[0]  # number of annotations = number of frames
        
        # access the videos 
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f"Error: Unable to open video file '{video_path}'.")
            continue
        
        fps = capture.get(cv2.CAP_PROP_FPS) 
        if fps <= 0:
            print(f"Error: Invalid FPS for video '{video_path}'.")
            capture.release()
            continue
        
        # logging
        pbar = tqdm(total=total_frames, desc=f"Extracting frames from {video_file}")
        frame_nr = 0
        
        # iterate and extract frames to match annotation rows
        for i in range(total_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, i) 
            success, frame = capture.read()
            
            if success:
                # save frame as "<video>_frame<frame_number>.png"
                frame_name = f"{os.path.splitext(os.path.basename(video_file))[0]}_frame{i:03d}.png"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_nr += 1
            
            pbar.update(1)
        
        capture.release()
        pbar.close()
        print(f"Frames extracted from {video_file} and saved to {output_dir}.")


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
                    