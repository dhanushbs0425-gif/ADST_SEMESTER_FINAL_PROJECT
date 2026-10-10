"""
Unit Tests for SQLite Database Storage & Retrieval
"""

import os
import pytest

from backend.database import init_db, save_analysis, get_history, get_analysis_by_id

def test_database_crud():
    init_db()
    
    mock_result = {
        'original_image_path': 'test_seed.jpg',
        'annotated_image_filename': 'annotated_test.jpg',
        'primary_seed_name': 'Rice',
        'primary_condition': 'Healthy',
        'primary_defect_type': 'None',
        'primary_confidence': 0.965,
        'total_seeds': 1,
        'healthy_count': 1,
        'defective_count': 0,
        'healthy_percentage': 100.0,
        'species_distribution': {'Rice': 1},
        'defect_distribution': {},
        'model_used': 'DL',
        'predictions': [{
            'seed_name': 'Rice',
            'condition': 'Healthy',
            'defect_type': 'None',
            'confidence': 0.965
        }]
    }
    
    rec_id = save_analysis(mock_result)
    assert rec_id > 0
    
    history = get_history()
    assert len(history) >= 1
    
    fetched = get_analysis_by_id(rec_id)
    assert fetched is not None
    assert fetched['seed_name'] == 'Rice'
    assert fetched['total_seeds'] == 1
    assert len(fetched['seeds']) == 1
