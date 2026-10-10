"""
Unit Tests for Seed Image Preprocessing & Watershed Segmentation
"""

import numpy as np
import cv2
import pytest

from backend.segmentation import preprocess_image, segment_seeds

def test_preprocessing_and_segmentation():
    # Create synthetic tray image with 2 circles
    img = np.full((200, 200, 3), 240, dtype=np.uint8) # light background
    cv2.circle(img, (60, 60), 25, (40, 80, 150), -1) # Seed 1
    cv2.circle(img, (140, 140), 25, (40, 80, 150), -1) # Seed 2
    
    preprocessed = preprocess_image(img, target_size=(200, 200))
    assert preprocessed.shape == (200, 200, 3)
    
    crops, annotated = segment_seeds(img)
    assert len(crops) >= 1
    assert annotated.shape == img.shape
    assert 'crop' in crops[0]
    assert 'bbox' in crops[0]
