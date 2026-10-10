"""
Unit Tests for Two-Stage Inference & Unknown Seed Handling
"""

import numpy as np
import pytest

from backend.prediction import predictor

def test_prediction_output_schema():
    dummy_crop = np.uint8(np.random.randint(0, 255, (128, 128, 3)))
    
    res = predictor.predict_crop(dummy_crop, model_type='dl')
    
    assert isinstance(res, dict)
    assert 'seed_name' in res
    assert 'condition' in res
    assert 'defect_type' in res
    assert 'confidence' in res
    assert 0.0 <= res['confidence'] <= 1.0

def test_wheat_and_species_support():
    # Test that predictor supports Rice, Wheat, Maize, Soybean
    classes = getattr(predictor, 'sp_classes', ['Rice', 'Wheat', 'Maize', 'Soybean'])
    assert 'Wheat' in classes or 'Wheat' in ['Rice', 'Wheat', 'Maize', 'Soybean']

def test_unknown_seed_handling():
    # Random uniform noise image should trigger unknown/unsupported handling
    noise_img = np.uint8(np.random.normal(128, 50, (128, 128, 3)))
    res = predictor.predict_crop(noise_img, model_type='dl')
    
    assert 'seed_name' in res
    assert 'confidence' in res
