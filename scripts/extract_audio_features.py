import librosa
import numpy as np



def extract_zcr(audio_data):
    zcr_features = {}

    for video_idx, segments in audio_data.items():
        zcr_features[video_idx] = []
        for frame_idx, audio_path in segments:
            try:
                # Load audio segment
                audio, sr = librosa.load(audio_path, sr=None)

                # Check if audio is empty
                if len(audio) == 0:
                    print(f"Warning: Empty audio at {audio_path}. Skipping.")
                    continue

                # Calculate ZCR
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
                
                # Store ZCR value
                zcr_features[video_idx].append((frame_idx, zcr))
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    return zcr_features