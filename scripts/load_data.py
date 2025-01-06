import os
import pandas as pd
import librosa
import cv2
from tqdm import tqdm
from pathlib import Path
import importlib.util
import subprocess
import sys

from scripts.setup import run_setup

def check_extracted(frames_output_dir, audio_output_dir):
    """
    Check if frames and audio are extracted by verifying their respective directories.
    """
    frames_extracted = any(Path(frames_output_dir).glob("*.png"))
    audio_extracted = any(Path(audio_output_dir).glob("*.wav"))

    return frames_extracted, audio_extracted


def load_full_audio(audio_output_dir, muppet_files, data_path):
    """
    Load audio files into memory. - needs to be called manually if wanted
    """
    audio_data = []

    # First, map audio file names to video file names
    audio_to_video_map = {Path(video_file).stem: video_file for video_file in muppet_files.keys()}

    # Process audio files and add the video duration
    for audio_file in Path(audio_output_dir).glob("*.wav"):
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=None)
            audio_file_name = Path(audio_file).stem
            
            # Get the corresponding video file name
            video_file = audio_to_video_map.get(audio_file_name)
            if not video_file:
                raise ValueError(f"No corresponding video file found for audio file: {audio_file_name}")
            
            # Extract video duration
            video_path = os.path.join(data_path, video_file)
            duration_command = (
                f"ffprobe -v error -select_streams v:0 -show_entries format=duration "
                f"-of default=noprint_wrappers=1:nokey=1 \"{video_path}\""
            )
            ffprobe_out = os.popen(duration_command).read().strip()
            if not ffprobe_out:
                raise ValueError(f"Could not retrieve duration for video: {video_path}")
            
            video_duration = float(ffprobe_out)

            # Append the data with the duration
            audio_data.append({
                "audio_file": audio_file.name,
                "audio": audio,
                "sr": sr,
                "duration": video_duration
            })
        except Exception as e:
            print(f"Error processing audio file '{audio_file}': {e}")
            continue



    # for audio_file in Path(audio_output_dir).glob("*.wav"):
    #     audio, sr = librosa.load(str(audio_file), sr=None)
    #     audio_data.append({"audio_file": audio_file.name, "audio": audio, "sr": sr})


    # for i in muppet_files.keys():
    #     video_path = os.path.join(data_path, i)
        
    #     try:
    #         # extract video duration
    #         duration_command = (
    #             f"ffprobe -v error -select_streams v:0 -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{video_path}\""
    #         )

    #         ffprobe_out = os.popen(duration_command).read().strip()
    #         # Check if output is empty
    #         if not ffprobe_out:
    #             raise ValueError(f"Could not retrieve duration for video: {video_path}")

    #     except Exception as e:
    #         print(f"Error retrieving duration for video '{i}': {e}")
    #         continue
    #     video_duration = float(ffprobe_out)    
    
    
    
    print(f"Loaded {len(audio_data)} audio files.")

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



def check_and_load(data_path, frames_output_dir, audio_output_dir, annotations_path, muppet_files):
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
        #run_setup_script(data_path, frames_output_dir, audio_output_dir, annotations_path)
        run_setup(data_path, frames_output_dir, audio_output_dir, annotations_path, muppet_files)
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
    audio_data = load_full_audio(audio_output_dir, muppet_files, data_path)

    print(f"Loaded audio segments for {len(audio_data)} videos.")
    # Inspect loaded audio segments
    # for video_idx, segment_list in audio_data.items():
    #     print(f"Video {video_idx} has {len(segment_list)} audio segments.")

    # Load and inspect frames
    frames = load_frames(muppet_files, frames_output_dir)
    print(f"Number of videos with frames: {len(frames)}")
    for video_idx, frame_list in frames.items():
        print(f"Video {video_idx} has {len(frame_list)} frames.")


    

    return annotations, audio_data, frames
    

