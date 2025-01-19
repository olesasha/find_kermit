import os
import pandas as pd
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv
from tqdm import tqdm
from pathlib import Path
from scipy.fftpack import dct
from skimage.util import img_as_ubyte

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






def extract_hsv_features(frames, bins=16, output_path = '../model_vars/sim2_video/hsv_feature_df.csv', save_df = True):
    """
    Extract color histograms from video frames in HSV color space.

    Parameters:
    - frames (dict): A dictionary mapping video indices to a list of (frame_idx, frame_path).
    - output_path (str): Path to save the extracted color histogram features.
    - bins (int): Number of bins for each color channel histogram.

    Returns:
    - color_features_df (pd.DataFrame): DataFrame containing color histogram features for each frame.
    """
    color_features = []

    for video_idx, frame_list in tqdm(frames.items(), desc="Extracting Color Histograms"):
        for frame_idx, frame_path in frame_list:
            try:
                # Read the frame
                frame = imread(frame_path)

                # Convert to HSV color space
                hsv_frame = rgb2hsv(frame)

                # Compute histograms for each channel
                hist_features = []
                for channel in range(hsv_frame.shape[-1]):
                    hist, _ = np.histogram(
                        hsv_frame[..., channel], bins=bins, range=(0, 1), density=True
                    )
                    hist_features.extend(hist)

                # Create a feature row
                feature_row = {f"hsv_channel_{i}_bin_{j}": hist_features[i * bins + j]
                               for i in range(hsv_frame.shape[-1]) for j in range(bins)}
                feature_row.update({
                    "video_idx": video_idx,
                    "frame_idx": frame_idx
                })

                color_features.append(feature_row)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")

    # Convert to DataFrame
    color_features_df = pd.DataFrame(color_features)

    # Save to output path
    if save_df:
        color_features_df.to_csv(output_path, index=False)
        print(f"Color histogram features saved to {output_path}")

    return color_features_df


###########################################################
###########################################################
###########################################################

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv
from scipy.fftpack import dct
from skimage.util import img_as_ubyte

# Constants for LBP
LBP_RADIUS = 1  # Radius of the circular LBP pattern
LBP_POINTS = 8 * LBP_RADIUS  # Number of points in the circular pattern
LBP_METHOD = "uniform"  # Uniform LBP pattern

def visualize_lbp(frame):
    gray_frame = rgb2gray(frame)
    gray_frame = img_as_ubyte(gray_frame)
    lbp = local_binary_pattern(gray_frame, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    
    # Increase the figure size
    plt.figure(figsize=(14, 7))  # Width=14 inches, Height=7 inches
    
    plt.subplot(1, 2, 1)
    plt.title("Original Frame (Grayscale)", fontsize=14)
    plt.imshow(gray_frame, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("LBP Features", fontsize=14)
    plt.imshow(lbp, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def visualize_dct(frame, dct_size=8):
    gray_frame = rgb2gray(frame)
    dct_transform = dct(dct(gray_frame, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_block = dct_transform[:dct_size, :dct_size]

    # Increase figure size
    plt.figure(figsize=(16, 8))  # Larger width and height for better visualization
    
    plt.subplot(1, 2, 1)
    plt.title("Original Frame (Grayscale)", fontsize=14)
    plt.imshow(gray_frame, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"DCT (Top {dct_size}x{dct_size} Coefficients)", fontsize=14)
    plt.imshow(dct_block, cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

    
def visualize_hsv(frame, bins=16):
    hsv_frame = rgb2hsv(frame)
    channels = ['Hue', 'Saturation', 'Value']

    # Increase figure size for the HSV channels
    plt.figure(figsize=(18, 6))  # Wider figure for three subplots side-by-side
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(f"{channels[i]} Channel", fontsize=14)
        plt.imshow(hsv_frame[..., i], cmap="hsv" if i == 0 else "gray")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

    # Plot histograms with a larger figure size
    histograms = [np.histogram(hsv_frame[..., i], bins=bins, range=(0, 1), density=True)[0] for i in range(3)]
    plt.figure(figsize=(14, 7))  # Larger figure for the histograms
    for i, hist in enumerate(histograms):
        plt.plot(hist, label=f"{channels[i]} Histogram", linewidth=2)
    plt.title("HSV Histograms", fontsize=14)
    plt.xlabel("Bins", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
