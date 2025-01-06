import librosa
import numpy as np
from scipy.signal import correlate


# Define constants
AUDIO_SAMPLING_RATE = 44100
FRAMES_PER_SECOND = 25
# Calculate expected length
samples_per_frame = int(AUDIO_SAMPLING_RATE / FRAMES_PER_SECOND)

def pad_audio(audio, duration):
    """
    Pad audio with zeros to match the expected length.

    Parameters:
    - audio (ndarray): Audio signal.
    - expected_length (int): Required audio length in samples.

    Returns:
    - ndarray: Padded audio signal.
    """
    req_length = int(AUDIO_SAMPLING_RATE * duration)
    #print(f"exp: {req_length}")
    #print(len(audio))
    if len(audio) < req_length:
        padding = req_length - len(audio)
        audio = np.pad(audio, (0, padding - 1), mode="constant")
        #print(padding)

    return audio

def extract_zcr(audio_data):
    """
    Extract ZCR (Zero-Crossing Rate) features for each audio file.

    Parameters:
    - audio_data (list): List of dictionaries with audio data and metadata.

    Returns:
    - zcr_features (list): List of ZCR features for each audio file.
    """
    zcr_features = []

    for audio_entry in audio_data:
        try:
            audio = audio_entry['audio']
            duration = audio_entry['duration']

            # Pad and normalize audio
            audio = pad_audio(audio, duration)
            audio = librosa.util.normalize(audio)

            # Calculate ZCR
            zcr = librosa.feature.zero_crossing_rate(
                    y=audio,
                    hop_length=samples_per_frame,
                    frame_length=samples_per_frame,
                )[0]

            zcr_features.append(zcr)
        except Exception as e:
            print(f"Error processing {audio_entry['audio_file']}: {e}")
            zcr_features.append(None)

    return zcr_features

def extract_loudness(audio_data):
    """
    Extract loudness (RMS) features for each audio file.

    Parameters:
    - audio_data (list): List of dictionaries with audio data and metadata.

    Returns:
    - loudness_features (list): List of loudness features for each audio file.
    """
    loudness_features = []

    for audio_entry in audio_data:
        try:
            audio = audio_entry['audio']
            duration = audio_entry['duration']

            # Pad and normalize audio
            audio = pad_audio(audio, duration)
            audio = librosa.util.normalize(audio)

            # Calculate RMS
            rms = librosa.feature.rms(
                y=audio,
                frame_length=samples_per_frame,
                hop_length=samples_per_frame
            )[0]

            loudness_features.append(rms)
        except Exception as e:
            print(f"Error processing {audio_entry['audio_file']}: {e}")
            loudness_features.append(None)

    return loudness_features

def extract_rhythm(audio_data):
    """
    Extract rhythm features using autocorrelation for each frame of audio files.

    Parameters:
    - audio_data (list): List of dictionaries with audio data and metadata.

    Returns:
    - rhythm_features (list): List of rhythm features for each audio file.
    """
    rhythm_features = []

    for audio_entry in audio_data:
        try:
            audio = audio_entry['audio']
            duration = audio_entry['duration']

            # Pad and normalize audio
            audio = pad_audio(audio, duration)
            audio = librosa.util.normalize(audio)

            # Split audio into frames
            frames = librosa.util.frame(audio, frame_length=samples_per_frame, hop_length=samples_per_frame)

            # Calculate rhythm strength for each frame
            frame_rhythms = []
            for frame in frames.T:
                # Calculate autocorrelation
                autocorr = correlate(frame, frame, mode="full")
                autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags
                autocorr /= np.max(autocorr) if np.max(autocorr) != 0 else 1

                # Extract rhythm strength
                rhythm_strength = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
                frame_rhythms.append(rhythm_strength)

            rhythm_features.append(np.array(frame_rhythms))
        except Exception as e:
            print(f"Error processing {audio_entry['audio_file']}: {e}")
            rhythm_features.append(None)

    return rhythm_features


def create_target_variable(df):
    """
    Create a target variable with:
    0: Neither Kermit nor Waldorf & Statler are present.
    1: Only Kermit is present.
    2: Only Waldorf & Statler are present.
    3: Both are present.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'Kermit' and 'Audio_StatlerWaldorf' columns.

    Returns:
    - pd.Series: Target variable.
    """
    # Create the target variable based on the specified conditions
    target = (
        (df['Kermit'] == 1).astype(int) * 1 +  # Kermit contributes 1
        (df['Audio_StatlerWaldorf'] == 1).astype(int) * 2  # Statler & Waldorf contribute 2
    )
    return target
