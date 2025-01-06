import os
import pandas as pd
import librosa
import cv2
from tqdm import tqdm
import soundfile as sf
import argparse

def load_annotations(data_path: str, muppet_files: dict):
    """
    Load annotation files for the videos.
    
    Parameters:
    - data_path (str): Directory containing the annotation files.
    - muppet_files (dict): Dictionary mapping video file names to annotation file paths.
    
    Returns:
    - dict: A dictionary mapping video file names to loaded annotation DataFrames.
    """
    annotations = {}
    for video_file, annotation_file in muppet_files.items():
        annotation_path = os.path.join(data_path, annotation_file)
        if os.path.exists(annotation_path):
            annotations[video_file] = pd.read_csv(annotation_path, sep=";")
            print(f"Loaded annotations for '{video_file}'.")
        else:
            print(f"Error: Annotation file '{annotation_path}' not found.")
    return annotations

def extract_frames(muppet_files: dict, annotations: dict, data_path: str, output_dir: str):
    """
    Extract frames from videos and save them as PNG files.
    
    Parameters:
    - muppet_files (dict): Dictionary mapping video file names to annotation file paths.
    - annotations (dict): Dictionary mapping video file names to annotation DataFrames.
    - data_path (str): Directory containing the video files.
    - output_dir (str): Directory to save the extracted frames.
    """
    #output_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    for video_file, annotation_df in annotations.items():
        video_path = os.path.join(data_path, video_file)
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            continue

        # Open video
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f"Error: Unable to open video file '{video_path}'.")
            continue

        total_frames = len(annotation_df)  # Number of frames = number of annotations
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"Error: Invalid FPS for video '{video_path}'.")
            capture.release()
            continue

        # Extract frames
        pbar = tqdm(total=total_frames, desc=f"Extracting frames from {video_file}")
        for i in range(total_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = capture.read()
            if success:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame{i:03d}.png"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
            pbar.update(1)

        capture.release()
        pbar.close()
        print(f"Frames extracted from '{video_file}' and saved to '{output_dir}'.")


def extract_audio(muppet_files: dict, data_path: str, output_dir: str, sampling_rate: int = 44100):
    """
    Extract full audio from video files and save them in the output directory.

    Parameters:
    - muppet_files (dict): Dictionary mapping video filenames to annotation filenames (annotations ignored here).
    - data_path (str): Directory containing the video files.
    - output_dir (str): Directory to save the extracted audio files.
    - sampling_rate (int): The desired sampling rate for the extracted audio.
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_file in muppet_files.keys():
        video_path = os.path.join(data_path, video_file)
        output_audio_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + ".wav")

        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            continue

        try:
            # Extract audio using ffmpeg
            command = f'ffmpeg -i "{video_path}" -ab 160k -ac 2 -ar {sampling_rate} -vn "{output_audio_path}"'
            os.system(command)
            print(f"Audio extracted for '{video_file}' and saved to '{output_audio_path}'.")
        except Exception as e:
            print(f"Error extracting audio for '{video_file}': {e}")



def run_setup(data_path, frames_output_dir, audio_output_dir, annotations_path, muppet_files):
    # Step 1: Load annotations
    annotations = load_annotations(annotations_path, muppet_files)

    # Step 2: Extract frames
    extract_frames(muppet_files, annotations, data_path, frames_output_dir)

    # Step 3: Extract audio
    extract_audio(muppet_files, data_path, audio_output_dir)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Extract frames and audio for Muppet videos.")
#     parser.add_argument("--data_path", type=str, required=True, help="Directory containing the video files.")
#     parser.add_argument("--frames_output_dir", type=str, required=True, help="Directory to save extracted frames.")
#     parser.add_argument("--audio_output_dir", type=str, required=True, help="Directory to save extracted audio.")
#     parser.add_argument("--annotations_path", type=str, required=True, help="Directory containing annotation files.")
#     args = parser.parse_args()

#     data_path = args.data_path
#     frames_output_dir = args.frames_output_dir
#     audio_output_dir = args.audio_output_dir
#     annotations_path = args.annotations_path

#     # TODO: das hier ist nicht elegant!
#     muppet_files = {
#         "Muppets-02-01-01.avi": "GroundTruth_Muppets-02-01-01.csv",
#         "Muppets-02-04-04.avi": "GroundTruth_Muppets-02-04-04.csv",
#         "Muppets-03-04-03.avi": "GroundTruth_Muppets-03-04-03.csv",
#     }

#     # Step 1: Load annotations
#     annotations = load_annotations(annotations_path, muppet_files)

#     # Step 2: Extract frames
#     extract_frames(muppet_files, annotations, data_path, frames_output_dir)

#     # Step 3: Extract audio
#     extract_audio(muppet_files, data_path, audio_output_dir, annotations_path)

