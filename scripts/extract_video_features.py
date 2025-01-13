import os
import pandas as pd
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray
from tqdm import tqdm
from pathlib import Path
from scipy.fftpack import dct


# Constants for LBP
LBP_RADIUS = 1  # Radius of the circular LBP pattern
LBP_POINTS = 8 * LBP_RADIUS  # Number of points in the circular pattern
LBP_METHOD = "uniform"  # Uniform LBP pattern
from skimage.util import img_as_ubyte

def extract_lbp_features(frames, output_path = '../model_vars/sim2_video/lbp_feature_df.csv', save_df = True):
    """
    Extract LBP features from video frames.

    Parameters:
    - frames (dict): A dictionary mapping video indices to a list of (frame_idx, frame_path).
    - output_path (sb tr): Path to save the extracted LBP features.

    Returns:
    - lbp_features_df (pd.DataFrame): DataFrame containing LBP features for each frame.
    """
    # List to store feature rows
    lbp_features = []

    for video_idx, frame_list in tqdm(frames.items(), desc="Extracting LBP features"):
        for frame_idx, frame_path in frame_list:
            try:
                # Read and convert the frame to grayscale
                frame = imread(frame_path)
                gray_frame = rgb2gray(frame)

                # Convert the grayscale image to uint8
                gray_frame = img_as_ubyte(gray_frame)

                # Compute LBP for the grayscale image
                lbp = local_binary_pattern(gray_frame, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)

                # Calculate histogram of LBP values
                lbp_hist, _ = np.histogram(
                    lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2), density=True
                )

                # Create a feature row
                feature_row = {f"lbp_bin_{i}": lbp_hist[i] for i in range(len(lbp_hist))}
                feature_row.update({
                    "video_idx": video_idx,
                    "frame_idx": frame_idx
                })

                lbp_features.append(feature_row)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")

    # Convert to DataFrame
    lbp_features_df = pd.DataFrame(lbp_features)

    # Save to output path
    if save_df:
        lbp_features_df.to_csv(output_path, index=False)
        print(f"LBP features saved to {output_path}")

    return lbp_features_df



def extract_dct_features(frames, dct_size=8, output_path = '../model_vars/sim2_video/dct_feature_df.csv',  save_df = True):
    """
    Extract DCT features from video frames.

    Parameters:
    - frames (dict): A dictionary mapping video indices to a list of (frame_idx, frame_path).
    - output_path (str): Path to save the extracted DCT features.
    - dct_size (int): Size of the DCT block (e.g., 8 for an 8x8 block).

    Returns:
    - dct_features_df (pd.DataFrame): DataFrame containing DCT features for each frame.
    """
    # List to store feature rows
    dct_features = []

    for video_idx, frame_list in tqdm(frames.items(), desc="Extracting DCT features"):
        for frame_idx, frame_path in frame_list:
            try:
                # Read and convert the frame to grayscale
                frame = imread(frame_path)
                gray_frame = rgb2gray(frame)

                # Compute the DCT for the grayscale image
                dct_transform = dct(dct(gray_frame, axis=0, norm='ortho'), axis=1, norm='ortho')

                # Retain only the top-left dct_size x dct_size block
                dct_block = dct_transform[:dct_size, :dct_size].flatten()

                # Create a feature row
                feature_row = {f"dct_coeff_{i}": dct_block[i] for i in range(len(dct_block))}
                feature_row.update({
                    "video_idx": video_idx,
                    "frame_idx": frame_idx
                })

                dct_features.append(feature_row)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")

    # Convert to DataFrame
    dct_features_df = pd.DataFrame(dct_features)

    # Save to output path
    if save_df:
        dct_features_df.to_csv(output_path, index=False)
        print(f"DCT features saved to {output_path}")

    return dct_features_df