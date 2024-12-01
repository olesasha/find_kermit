import re
import os 
import cv2

def extract_frames(data_path:str, start_time_sec:int=15):
    """
    Extract frames from a video, skipping the first `start_time_sec` seconds.
    - data_path (str): Path to the video file.
    - start_time_sec (int): Time in seconds to skip from the start of the video.
    """
    capture = cv2.VideoCapture(data_path)
    capture.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    
    success, frame = capture.read()
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)
    frame_nr = 0

    while success:
        cv2.imwrite(f'../ground_truth_data/frame{frame_nr}.png', frame)

        success, frame = capture.read()
        frame_nr += 1
        pbar.update(1)

    capture.release()
    pbar.close()
    print("Frames extracted successfully, skipping the first", start_time_sec, "seconds")

def train_test_split(frames_path:str, train_path:str, val_path:str, test_path:str):
    """
    Splits the frames within repository into train, validation and test split.
    Removes frames from the origin directory. 
    - frames_path (str): origin directory with frames. 
    - train_path, val_path, test_path (str): target directories. 
    """
    print("tbd")
    
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
                    