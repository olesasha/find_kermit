
import librosa
import numpy as np
from scipy.signal import correlate
from tqdm import tqdm


# Define constants
AUDIO_SAMPLING_RATE = 44100
FRAMES_PER_SECOND = 25
# Calculate expected length
samples_per_frame = int(AUDIO_SAMPLING_RATE / FRAMES_PER_SECOND)

# -----------
# The functions pad_audio and extract_mfcc are also part of the SIM1 submission
# -----------

def pad_audio(audio, duration):
    """
    Pad audio with zeros to match the expected length.

    Parameters:
    - audio (ndarray): audio signal.
    - expected_length (int): required audio length in samples.

    Returns:
    - ndarray: padded audio signal.
    """
    req_length = int(AUDIO_SAMPLING_RATE * duration)

    if len(audio) < req_length:
        padding = req_length - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio

def extract_mfcc(audio_data, n_mfcc=20):
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs) for each audio file.

    Parameters:
    - audio_data (list): List of dictionaries with audio data and metadata.
    - n_mfcc (int): Number of MFCCs to extract.
    - frames_per_second (int): Frame rate for MFCC extraction.

    Returns:
    - mfcc_features (list): List of MFCC features for each audio file.
    """
    mfcc_features = []

    for audio_entry in audio_data:
        try:
            audio = audio_entry['audio']
            sr = audio_entry['sr']
            duration = audio_entry['duration']

            # Pad and normalize audio
            audio = pad_audio(audio, duration)
            audio = librosa.util.normalize(audio)

            # Compute MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                hop_length=samples_per_frame,
                win_length=samples_per_frame,
                n_mfcc=n_mfcc,
                n_fft=samples_per_frame
            )

            mfcc_features.append(mfcc)
        except Exception as e:
            print(f"Error processing {audio_entry['audio_file']}: {e}")
            mfcc_features.append(None)

    return mfcc_features


def extract_log_mel_spectrogram(audio_data, n_mels=40, frames_per_second=25):
    """
    Extracts log-Mel spectrograms from a list of audio data entries.

    Parameters:
        audio_data (list): list of dictionaries with audio data and metadata.
        n_mels (int): number of Mel bands to generate.
        frames_per_second (int): target frames per second for spectrogram.

    Returns:
        log_mel_features list: Log-Mel spectrograms or None for entries with errors.
    """

    log_mel_features = []

    for audio_entry in audio_data:
        try:
            audio = audio_entry['audio']
            sr = audio_entry['sr']
            duration = audio_entry['duration']

            # Pad and normalize audio
            audio = pad_audio(audio, duration)
            audio = librosa.util.normalize(audio)

            # Compute Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=samples_per_frame, 
                hop_length=samples_per_frame, 
                n_mels=n_mels
            )

            # Convert to log scale
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

            log_mel_features.append(log_mel_spectrogram)
        except Exception as e:
            print(f"Error processing {audio_entry['audio_file']}: {e}")
            log_mel_features.append(None)

    return log_mel_features


def extract_yin(audio_data, frames_per_second=25):
    """
    Extracts YIN-based pitch feature from a list of audio data.

    Parameters:
        audio_data (list): list of dictionaries with audio data and metadata.
        frames_per_second (int): target frame rate for pitch extraction.

    Returns:
        yin_features dict: a dictionary where keys are audio file names, and values are lists containing 
              YIN pitch estimates, voiced flags, and voiced probabilities. Returns None for entries with errors.
    """
    
    yin_features = {}

    # wrap audio_data with tqdm for a progress bar
    for audio_entry in tqdm(audio_data, desc="Processing audio entries"):
        try:
            audio = audio_entry['audio']
            sr = audio_entry['sr']
            duration = audio_entry['duration']
            audio_file = audio_entry.get('audio_file', 'Unknown file')

            # pad and normalize audio
            audio = pad_audio(audio, duration)
            audio = librosa.util.normalize(audio)

            # number of samples per frame
            samples_per_frame = sr // frames_per_second
            
            # yin pitch extraction
            yin, voiced_flag, voiced_probs = librosa.pyin(
                y=audio,
                sr=sr,
                hop_length=samples_per_frame,  
                fmin=librosa.note_to_hz('C1'),
                fmax=librosa.note_to_hz('C8'))

            # Log details for each frame using tqdm.write
            # for frame_idx, (pitch, voiced, prob) in enumerate(zip(yin, voiced_flag, voiced_probs)):
            #    tqdm.write(
            #    f"File: {audio_file}, Frame: {frame_idx}, "
            #    f"Pitch: {pitch}, Voiced: {voiced}, Probability: {prob}")

            yin_features[audio_file] = [yin, voiced_flag, voiced_probs]
             
        except Exception as e:
            tqdm.write(f"Error processing {audio_entry.get('audio_file', 'Unknown file')}: {e}")
            yin_features[audio_file] = None

    return yin_features
