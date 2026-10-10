"""
Unit Tests for Dataset Setup and Report Verification
"""

import os
import json
import pytest

from backend.config import BASE_DIR, DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR

def test_dataset_directories_exist():
    assert os.path.exists(DATASET_DIR)
    assert os.path.exists(TRAIN_DIR)
    assert os.path.exists(VAL_DIR)
    assert os.path.exists(TEST_DIR)

def test_dataset_report_json():
    report_path = os.path.join(DATASET_DIR, 'dataset_report.json')
    assert os.path.exists(report_path)
    with open(report_path, 'r') as f:
        data = json.load(f)
    assert 'total_unified_images' in data
    assert data['total_unified_images'] > 0
    assert 'supported_seed_types' in data
    assert len(data['supported_seed_types']) >= 3
