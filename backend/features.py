"""
SeedSense AI - Feature Extraction Module
Extracts Color, Shape, and Texture (GLCM) features from seed images for Machine Learning.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_color_features(img_bgr):
    """
    Extracts RGB and HSV mean and standard deviation features (12 features).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    rgb_mean = np.mean(img_rgb, axis=(0, 1))
    rgb_std = np.std(img_rgb, axis=(0, 1))
    
    hsv_mean = np.mean(img_hsv, axis=(0, 1))
    hsv_std = np.std(img_hsv, axis=(0, 1))
    
    features = np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std])
    return features

def extract_shape_features(img_bgr):
    """
    Extracts area, perimeter, width, height, aspect ratio, circularity, extent, solidity (8 features).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Default fallback shape features
        h, w = gray.shape
        return np.array([w * h, 2 * (w + h), w, h, w / float(h), 1.0, 1.0, 1.0])
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    
    aspect_ratio = float(w) / h if h > 0 else 1.0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    extent = float(area) / (w * h) if (w * h) > 0 else 0.0
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    
    return np.array([area, perimeter, w, h, aspect_ratio, circularity, extent, solidity])

def extract_texture_features(img_bgr):
    """
    Extracts GLCM contrast, homogeneity, energy, and correlation across 4 angles (16 features).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Quantize gray to 32 levels for fast GLCM computation
    gray_q = (gray // 8).astype(np.uint8)
    
    glcm = graycomatrix(gray_q, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    contrast_std = graycoprops(glcm, 'contrast').std()
    homogeneity_std = graycoprops(glcm, 'homogeneity').std()
    energy_std = graycoprops(glcm, 'energy').std()
    correlation_std = graycoprops(glcm, 'correlation').std()
    
    props = [
        contrast, homogeneity, energy, correlation,
        contrast_std, homogeneity_std, energy_std, correlation_std
    ]
    
    # Extra statistical texture features
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    var_val = np.var(gray)
    skewness = np.mean(((gray - mean_val) / (std_val + 1e-5)) ** 3)
    kurtosis = np.mean(((gray - mean_val) / (std_val + 1e-5)) ** 4)
    
    texture = np.array(props + [mean_val, std_val, var_val, skewness, kurtosis])
    return texture

def extract_all_features(img_bgr):
    """
    Extracts complete 33-dimensional feature vector (Color + Shape + Texture).
    """
    color_feats = extract_color_features(img_bgr)
    shape_feats = extract_shape_features(img_bgr)
    texture_feats = extract_texture_features(img_bgr)
    
    features = np.concatenate([color_feats, shape_feats, texture_feats])
    return features
