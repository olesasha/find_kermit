import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# HELPER FUNCTIONS 

def extract_dominant_colors(image_path, num_clusters=10, num_colors=5):
    """
    Extracts the dominant colors from an input image using k-means clustering. 
    Parameters
    ----------
    image : input image path
    num_clusters : number of clusters to use in the k-means algorithm.
                   This controls how finely the color space is partitioned.
    num_colors : number of dominant colors to return (must be equal to or less than num_clusters)
    
    Returns
    -------
    dominant_colors: a numpy array where each row represents a dominant color in HSV format.
    """
    
    image = cv2.imread(image_path)
    
    # convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # resize image to reduce computational complexity
    data = cv2.resize(hsv_image, (100, 100)).reshape(-1, 3)
    
    # stopping criteria: max_iter, epsilon (accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    
    # apply k-means clustering
    _, _, centers = cv2.kmeans(data.astype(np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    dominant_colors = centers[:num_colors]
    
    return dominant_colors


def extract_contours(image_path, d = 7):
    
    """
    Extracts edges from an image using bilateral filtering and Canny edge detection.

    Parameters
    ----------
    image : input image path
    d : Diameter of the pixel neighborhood used for bilateral filtering.
        If d <= 0, it is computed based on sigmaSpace.

    Returns
    -------
    Flattened array of edge-detected pixels, reshaped for classifier input.
    """
    
    image = cv2.imread(image_path)

    # convert the image to grayscale and resize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    
    # apply bilateral blur
    bilateral_blur = cv2.bilateralFilter(gray, d, 75, 75)

    # apply edge detector
    edges = cv2.Canny(bilateral_blur, 75, 75)
    
    #reshape for classifier input
    edges = edges.reshape(-1)

    return edges


def extract_glcm_features(image_path, distance=1, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract and normalize GLCM features from an image.

    Parameters:
    - image_path: input image path
    - distance: Distance for GLCM calculation (set to 1 for simplicity).
    - angles: List of angles for GLCM calculation.

    Returns:
    - A dictionary with normalized GLCM features.
    """
    
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_gray = cv2.equalizeHist(image_gray)  # histogram equalization for contrast enhancement
    image_gray = cv2.GaussianBlur(image_gray, (7, 7), 0)  # Gaussian blur to reduce noise

    # compute the GLCM features
    glcm = graycomatrix(image_gray, distances=[distance], angles=angles, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')
    correlation = graycoprops(glcm, 'correlation')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')

    # flatten the arrays
    contrast_flat = contrast.flatten()
    correlation_flat = correlation.flatten()
    energy_flat = energy.flatten()
    homogeneity_flat = homogeneity.flatten()

    # normalize the features using MinMaxScaler 
    scaler = MinMaxScaler()
    features = np.array([contrast_flat, correlation_flat, energy_flat, homogeneity_flat]).T
    features_normalized = scaler.fit_transform(features)

    # create a dictionary to store the normalized features
    feature_dict = {
        'Contrast': np.mean(features_normalized[:, 0]),
        'Correlation': np.mean(features_normalized[:, 1]),
        'Energy': np.mean(features_normalized[:, 2]),
        'Homogeneity': np.mean(features_normalized[:, 3])
    }

    return feature_dict

# EXTRACTION FUNCTIONS 
def extract_dominant_colors_from_video(frames, num_clusters=10, num_colors=5):
    """
    Extract dominant colors from video frames provided in a dictionary format.

    Parameters
    ----------
    frames : frames dict
    num_clusters : number of clusters to use in k-means for color extraction.
    num_colors : number of dominant colors to show. 

    Returns
    -------
    list of dicts where each dictionary contains video_id, frame_id and the dominant colors for that frame.
    """
    results = []

    for video_id, frame_list in frames.items():
        print(f"Processing video ID: {video_id}")
        for frame_id, frame_path in tqdm(frame_list, desc=f"Processing frames for video {video_id}"):
            try:
                dominant_colors = extract_dominant_colors(frame_path, num_clusters, num_colors)
                results.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "dominant_colors": dominant_colors.tolist()  
                })
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                results.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "dominant_colors": None 
                })

    return results

def extract_contours_from_video(frames, d=7):
    """
    Extract contours from video frames provided in a dictionary format.

    Parameters
    ----------
    frames : frames dict
    d: diameter of the pixel neighborhood used for bilateral filtering
    
    Returns
    -------
    list of dicts where each dictionary contains video_id, frame_id and the contours for that frame.
    """
    results = []

    for video_id, frame_list in frames.items():
        print(f"Processing video ID: {video_id}")
        for frame_id, frame_path in tqdm(frame_list, desc=f"Processing frames for video {video_id}", mininterval=2):
            try:
                contours = extract_contours(frame_path, d=7)
                results.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "contours": contours.tolist()  
                })
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                results.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "contours": None 
                })

    return results

def extract_glcm_features_from_video(frames, distance=1, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract GLCM features from video frames provided in a dictionary format.

    Parameters
    ----------
    frames : frames dict
    distance : distance for GLCM calculation
    angles : list of angles for GLCM calculation, by default [0, np.pi/4, np.pi/2, 3*np.pi/4].

    Returns
    -------
    list of dict containing video_id, frame_id, and the GLCM features for that frame.
    """
    results = []

    for video_id, frame_list in frames.items():
        print(f"Processing video ID: {video_id}")
        for frame_id, frame_path in tqdm(frame_list, desc=f"Processing frames for video {video_id}", mininterval=1):
            try:
                feature_dict = extract_glcm_features(frame_path, distance=distance, angles=angles)
                results.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    **feature_dict
                })
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                results.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "Contrast": None,
                    "Correlation": None,
                    "Energy": None,
                    "Homogeneity": None
                })

    return results
