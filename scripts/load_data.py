import os
import pandas as pd
import librosa
import cv2
from tqdm import tqdm
from pathlib import Path
import importlib.util
import subprocess
import sys

# Define paths
#data_path = "../ground_truth_data/trimmed_videos"
data_path = "../ground_truth_data"
frames_output_dir = "../ground_truth_data/frames"
audio_output_dir = "../ground_truth_data/audio"
#annotations_path = "../ground_truth_data/trimmed_videos"
annotations_path = "../ground_truth_data"

muppet_files = {
    "Muppets-02-01-01.avi": "GroundTruth_Muppets-02-01-01.csv",
    "Muppets-02-04-04.avi": "GroundTruth_Muppets-02-04-04.csv",
    "Muppets-03-04-03.avi": "GroundTruth_Muppets-03-04-03.csv",
}

# Flag to determine whether to load full audio or audio segments
LOAD_FULL_AUDIO = False

def check_extracted(frames_output_dir, audio_output_dir):
    """
    Check if frames and audio are extracted by verifying their respective directories.
    """
    frames_extracted = any(Path(frames_output_dir).glob("*.png"))
    audio_extracted = any(Path(audio_output_dir).glob("*.wav"))

    return frames_extracted, audio_extracted

def run_setup_script():
    """
    Runs the scripts/setup.py to extract frames and audio if needed.
    """
    setup_script = "scripts/setup.py"
    if not os.path.exists(setup_script):
        raise FileNotFoundError(f"{setup_script} not found. Please check the path")
    os.system(f"python {setup_script}")



def load_audio_segments(muppet_files: dict, audio_output_dir: str):
    """
    Load audio segments grouped by video filename prefix and sorted by frame index.

    Parameters:
    - muppet_files (dict): Dictionary mapping video filenames to annotation filenames.
    - audio_output_dir (str): Directory containing extracted audio segments.

    Returns:
    - dict: A dictionary where keys are video indices and values are lists of tuples (frame_idx, audio_path).
    """
    audio_segments = {}

    for video_idx, video_file in enumerate(muppet_files.keys()):
        video_prefix = os.path.splitext(video_file)[0]

        # Find all audio segments matching the prefix
        segment_files = [
            segment for segment in os.listdir(audio_output_dir)
            if segment.startswith(video_prefix) and segment.endswith(".wav")
        ]

        if not segment_files:
            print(f"Warning: No audio segments found for video {video_file}.")
            continue

        # Parse segment indices and create (frame_idx, audio_path) tuples
        ordered_segments = []
        for segment in segment_files:
            try:
                frame_idx = int(segment.split("_frame")[1].split(".")[0])  # Extract frame index
                segment_path = os.path.join(audio_output_dir, segment)
                ordered_segments.append((frame_idx, segment_path))
            except (IndexError, ValueError):
                print(f"Warning: Unable to parse frame index from '{segment}'. Skipping.")
                continue

        # Sort segments by frame index
        ordered_segments.sort(key=lambda tup: tup[0])
        audio_segments[video_idx] = ordered_segments

    print(f"Loaded audio segments for {len(audio_segments)} videos.")
    return audio_segments


def load_full_audio(audio_output_dir):
    """
    Load audio files into memory. - needs to be called manually if wanted
    """
    audio_data = []
    for audio_file in Path(audio_output_dir).glob("*.wav"):
        audio, sr = librosa.load(str(audio_file), sr=None)
        audio_data.append({"audio_file": audio_file.name, "audio": audio, "sr": sr})
    print(f"Loaded {len(audio_data)} audio files.")

    # Load audio based on the LOAD_FULL_AUDIO flag

    print("Loading full audio files...")
    full_audio_data = load_full_audio(os.path.join(audio_output_dir, "full_audio"))
    print(f"Loaded {len(audio_data)} full audio files.")
    for audio_file in full_audio_data:
        print(audio_file['audio_file'])

    return audio_data

def load_frames(muppet_files: dict, frames_output_dir: str):
    """
    Load frames grouped by video filename prefix and sorted by frame index.

    Parameters:
    - muppet_files (dict): Dictionary mapping video filenames to annotation filenames.
    - frames_output_dir (str): Directory containing extracted frames.

    Returns:
    - dict: A dictionary where keys are video indices and values are lists of tuples (frame_idx, frame_path).
    """
    frames = {}

    for video_idx, video_file in enumerate(muppet_files.keys()):
        # Use the video filename (without extension) as a prefix to identify frames
        video_prefix = os.path.splitext(video_file)[0]

        # Find all frames matching the prefix
        frame_files = [
            frame for frame in os.listdir(frames_output_dir)
            if frame.startswith(video_prefix) and frame.endswith(".png")
        ]

        if not frame_files:
            print(f"Warning: No frames found for video {video_file}.")
            continue

        # Parse frame indices and create (frame_idx, frame_path) tuples
        ordered_frames = []
        for frame in frame_files:
            try:
                frame_idx = int(frame.split("_frame")[1].split(".")[0])  # Extract frame index
                frame_path = os.path.join(frames_output_dir, frame)
                ordered_frames.append((frame_idx, frame_path))
            except (IndexError, ValueError):
                print(f"Warning: Unable to parse frame index from '{frame}'. Skipping.")
                continue

        # Sort frames by frame index
        ordered_frames.sort(key=lambda tup: tup[0])
        frames[video_idx] = ordered_frames

    print(f"Loaded frames for {len(frames)} videos.")
    return frames



def check_and_load():
    """
    Check if frames and audio are extracted, and load them along with annotations.

    Returns:
    - dict: Annotations loaded from the annotation files.
    - dict/list: Audio segments or full audio data based on LOAD_FULL_AUDIO.
    - dict: Frames grouped by video index, sorted by frame index.
    """
    # Check if frames and audio are extracted
    frames_extracted, audio_extracted = check_extracted(frames_output_dir, audio_output_dir)

    if not (frames_extracted and audio_extracted):
        print("Frames and/or audio not extracted. Running setup script...")
        run_setup_script()
    else:
        print("Frames and audio are already extracted.")

    # Load annotations
    annotations = {}
    for video_file, annotation_file in muppet_files.items():
        annotation_path = os.path.join(annotations_path, annotation_file)
        if os.path.exists(annotation_path):
            annotations[video_file] = pd.read_csv(annotation_path, sep=";")
        else:
            print(f"Annotation file '{annotation_file}' not found!")



    print("Loading audio segments...")
    audio_data = load_audio_segments(muppet_files, audio_output_dir)
    print(f"Loaded audio segments for {len(audio_data)} videos.")
    # Inspect loaded audio segments
    for video_idx, segment_list in audio_data.items():
        print(f"Video {video_idx} has {len(segment_list)} audio segments.")

    # Load and inspect frames
    frames = load_frames(muppet_files, frames_output_dir)
    print(f"Number of videos with frames: {len(frames)}")
    for video_idx, frame_list in frames.items():
        print(f"Video {video_idx} has {len(frame_list)} frames.")

    return annotations, audio_data, frames





if __name__ == "__main__":
    annotations, audio_data, frames = check_and_load()

    if LOAD_FULL_AUDIO:
        print("Full audio files loaded and ready for analysis.")
    
    
    print("Audio segments loaded and ready for analysis.")

    print("Frames and audio successfully loaded.")




