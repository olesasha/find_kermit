import os
import pandas as pd
import librosa
import cv2
from tqdm import tqdm
import soundfile as sf

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



def extract_audio(muppet_files: dict, data_path: str, output_dir: str, annotations_path: str):
    """
    Extract audio segments aligned to video frames based on ground truth data and save them in the output_dir.
    Save full audio files in a separate folder within the output directory.

    Parameters:
    - muppet_files (dict): Dictionary mapping video filenames to annotation filenames.
    - data_path (str): Base directory containing the video files.
    - output_dir (str): Directory to save the extracted audio segments.
    - annotations_path (str): Directory containing the ground truth annotation files.
    """
    # Ensure the output directory and subfolders exist
    os.makedirs(output_dir, exist_ok=True)
    full_audio_dir = os.path.join(output_dir, "full_audio")
    os.makedirs(full_audio_dir, exist_ok=True)

    # Loop through the muppet_files dictionary
    for video_file, annotation_file in muppet_files.items():
        video_path = os.path.join(data_path, video_file)
        annotation_path = os.path.join(annotations_path, annotation_file)

        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            continue

        if not os.path.exists(annotation_path):
            print(f"Error: Annotation file '{annotation_path}' not found.")
            continue

        # Load annotations
        annotations = pd.read_csv(annotation_path, sep=";")
        num_frames = len(annotations)

        # Retrieve video duration
        try:
            # extract video duration
            duration_command = (
                f"ffprobe -v error -select_streams v:0 -show_entries "
                f"format=duration -of default=noprint_wrappers=1:nokey=1 \"{video_path}\""
            )
            video_duration = float(os.popen(duration_command).read().strip())
        except Exception as e:
            print(f"Error retrieving duration for video '{video_file}': {e}")
            continue

        # Calculate number of audio samples per frame
        sampling_rate = 44100  # Standard audio sampling rate
        total_audio_samples = int(sampling_rate * video_duration)
        samples_per_frame = total_audio_samples // num_frames

        # Save full audio 
        full_audio_name = os.path.splitext(video_file)[0] + ".wav"
        full_audio_path = os.path.join(full_audio_dir, full_audio_name)

        try:
            os.system(
                f"ffmpeg -i \"{video_path}\" -ab 160k -ac 2 -ar {sampling_rate} -vn \"{full_audio_path}\""
            )
        except Exception as e:
            print(f"Error extracting full audio from '{video_file}': {e}")
            continue

        # Load the full audio
        try:
            audio, sr = librosa.load(full_audio_path, sr=sampling_rate)
        except Exception as e:
            print(f"Error loading audio from '{full_audio_path}': {e}")
            continue

        # Extract aligned audio segments
        for frame_idx in range(num_frames):
            start_sample = frame_idx * samples_per_frame
            end_sample = start_sample + samples_per_frame

            if end_sample > len(audio):
                end_sample = len(audio)  # Handle edge case for the last frame

            segment = audio[start_sample:end_sample]
            segment_name = f"{os.path.splitext(video_file)[0]}_frame{frame_idx:03d}.wav"
            segment_path = os.path.join(output_dir, segment_name)

            try:
                # Save the audio segment using soundfile
                sf.write(segment_path, segment, samplerate=sampling_rate)
            except Exception as e:
                print(f"Error writing audio segment {frame_idx} for '{video_file}': {e}")

        print(f"Extracted and aligned audio segments for '{video_file}'. Full audio saved to '{full_audio_dir}'.")


if __name__ == "__main__":
    # Define paths and files
    data_path = "../ground_truth_data"
    #data_path = "../ground_truth_data/trimmed_videos"
    
    frames_output_dir = "../ground_truth_data/frames"
    audio_output_dir = "../ground_truth_data/audio"
    #annotations_path = "../ground_truth_data/trimmed_videos"
    annotations_path = "../ground_truth_data"
    
    
    muppet_files = {
        "Muppets-02-01-01.avi": "GroundTruth_Muppets-02-01-01.csv",
        "Muppets-02-04-04.avi": "GroundTruth_Muppets-02-04-04.csv",
        "Muppets-03-04-03.avi": "GroundTruth_Muppets-03-04-03.csv",
    }

    # Step 1: Load annotations
    annotations = load_annotations(annotations_path, muppet_files)

    # Step 2: Extract frames
    extract_frames(muppet_files, annotations, data_path, frames_output_dir)

    # Step 3: Extract audio (pass annotations to avoid reloading them)
    extract_audio(muppet_files, data_path, audio_output_dir, annotations_path)

