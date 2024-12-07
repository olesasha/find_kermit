import os
import pandas as pd
import librosa
import cv2
from tqdm import tqdm


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

# TODO: vergiss nicht das auszukommentieren
# def extract_frames(muppet_files: dict, annotations: dict, data_path: str, output_dir: str):
#     """
#     Extract frames from videos and save them as PNG files.
    
#     Parameters:
#     - muppet_files (dict): Dictionary mapping video file names to annotation file paths.
#     - annotations (dict): Dictionary mapping video file names to annotation DataFrames.
#     - data_path (str): Directory containing the video files.
#     - output_dir (str): Directory to save the extracted frames.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     for video_file, annotation_df in annotations.items():
#         video_path = os.path.join(data_path, video_file)
#         if not os.path.exists(video_path):
#             print(f"Error: Video file '{video_path}' not found.")
#             continue

#         # Open video
#         capture = cv2.VideoCapture(video_path)
#         if not capture.isOpened():
#             print(f"Error: Unable to open video file '{video_path}'.")
#             continue

#         total_frames = len(annotation_df)  # Number of frames = number of annotations
#         fps = capture.get(cv2.CAP_PROP_FPS)
#         if fps <= 0:
#             print(f"Error: Invalid FPS for video '{video_path}'.")
#             capture.release()
#             continue

#         # Extract frames
#         pbar = tqdm(total=total_frames, desc=f"Extracting frames from {video_file}")
#         for i in range(total_frames):
#             capture.set(cv2.CAP_PROP_POS_FRAMES, i)
#             success, frame = capture.read()
#             if success:
#                 frame_name = f"{os.path.splitext(video_file)[0]}_frame{i:03d}.png"
#                 frame_path = os.path.join(output_dir, frame_name)
#                 cv2.imwrite(frame_path, frame)
#             pbar.update(1)

#         capture.release()
#         pbar.close()
#         print(f"Frames extracted from '{video_file}' and saved to '{output_dir}'.")


def extract_audio(muppet_files: dict, data_path: str, output_dir: str):
    """
    Extract audio from videos listed in the muppet_files dictionary and save them in the output_dir.

    Parameters:
    - muppet_files (dict): Dictionary mapping video filenames to annotation filenames.
    - data_path (str): Base directory containing the video files.
    - output_dir (str): Directory to save the extracted audio files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the muppet_files dictionary
    for video_file in tqdm(muppet_files.keys(), desc="Extracting audio from videos"):
        video_path = os.path.join(data_path, video_file)
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            continue

        # Define output audio file path
        audio_name = os.path.splitext(video_file)[0] + ".wav"
        audio_path = os.path.join(output_dir, audio_name)

        try:
            # FFmpeg command to extract audio
            os.system(
                # TODO: das hier cleaner
                f"C:\\ffmpeg\\bin\\ffmpeg.exe -i \"{video_path}\" -ab 160k -ac 2 -ar 44100 -vn \"{audio_path}\""
            )

            print(f"Extracted audio for '{video_file}' -> '{audio_path}'")
        except Exception as e:
            print(f"Error extracting audio from '{video_file}': {e}")

    print(f"All audio files saved to '{output_dir}'.")

    # TODO: load wav files


if __name__ == "__main__":
    # Define paths and files
    #data_path = "../ground_truth_data"
    data_path = "../ground_truth_data/trimmed_videos"
    frames_output_dir = "../ground_truth_data/frames"
    audio_output_dir = "../ground_truth_data/audio"
    muppet_files = {
        "Muppets-02-01-01.avi": "GroundTruth_Muppets-02-01-01.csv",
        "Muppets-02-04-04.avi": "GroundTruth_Muppets-02-04-04.csv",
        "Muppets-03-04-03.avi": "GroundTruth_Muppets-03-04-03.csv",
    }

    # Step 1: Load annotations
    annotations = load_annotations(data_path, muppet_files)

    # Step 2: Extract frames
    #extract_frames(muppet_files, annotations, data_path, frames_output_dir)

    # Step 3: Extract audio
    extract_audio(muppet_files, data_path, audio_output_dir)
