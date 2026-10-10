"""
Unit Tests for Color, Shape, and GLCM Texture Feature Extraction
"""

import numpy as np
import cv2
import pytest

from backend.features import (
    extract_color_features, extract_shape_features,
    extract_texture_features, extract_all_features
)

def test_feature_shapes():
    # Create dummy 64x64 BGR seed image
    dummy_img = np.uint8(np.random.randint(0, 255, (64, 64, 3)))
    
    color_feats = extract_color_features(dummy_img)
    shape_feats = extract_shape_features(dummy_img)
    texture_feats = extract_texture_features(dummy_img)
    all_feats = extract_all_features(dummy_img)
    
    assert len(color_feats) == 12
    assert len(shape_feats) == 8
    assert len(texture_feats) == 13
    assert len(all_feats) == 33
    assert not np.isnan(all_feats).any()
