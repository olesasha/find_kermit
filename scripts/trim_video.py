import os
from moviepy.video.io.VideoFileClip import VideoFileClip # pip install moviepy==2.0.0.dev2
from tqdm import tqdm

def trim_video_with_moviepy(input_video_path: str, output_video_path: str, duration: int = 60):
    """
    Trim a video to the first `duration` seconds, preserving both video and audio streams.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    try:
        print(f"Processing video: {input_video_path}")
        with VideoFileClip(input_video_path) as video:
            if video.duration < duration:
                print(f"Error: Video '{input_video_path}' is shorter than {duration} seconds.")
                return

            # Trim and save the video
            trimmed_video = video.subclip(0, duration)
            print(f"Saving trimmed video to: {output_video_path}")
            trimmed_video.write_videofile(
                output_video_path,
                codec="libx264",
                audio_codec="aac",
                #verbose=True,  # Enable verbose logs
                logger=None
            )
            print(f"Trimmed video saved successfully: {output_video_path}")
    except Exception as e:
        print(f"Error processing video '{input_video_path}': {e}")

# run
if __name__ == "__main__":
    data_path = "../ground_truth_data"
    trimmed_videos_dir = "../ground_truth_data/trimmed_videos"
    duration = 60

    muppet_files = [
        "Muppets-02-01-01.avi",
        "Muppets-02-04-04.avi",
        "Muppets-03-04-03.avi",
    ]

    for video_file in tqdm(muppet_files, desc="Trimming videos"):
        input_path = os.path.join(data_path, video_file)
        #output_path = os.path.join(trimmed_videos_dir, f"trimmed_{video_file}")
        output_path = os.path.join(trimmed_videos_dir, f"{video_file}")
        trim_video_with_moviepy(input_path, output_path, duration=duration)
