"""
Unit Tests for ReportLab PDF Generation
"""

import os
import pytest

from backend.config import RESULTS_DIR
from backend.report import generate_pdf_report

def test_pdf_report_generation():
    mock_data = {
        'id': 999,
        'date': '2026-10-03 12:00:00',
        'filename': 'sample_seed.jpg',
        'annotated_filename': 'annotated_result.jpg',
        'model_used': 'DL',
        'total_seeds': 5,
        'healthy_count': 4,
        'defective_count': 1,
        'healthy_percentage': 80.0,
        'primary_seed_name': 'Maize',
        'primary_condition': 'Healthy',
        'primary_defect_type': 'None',
        'primary_confidence': 0.94,
        'seeds': [
            {'seed_index': 1, 'seed_name': 'Maize', 'condition': 'Healthy', 'defect_type': 'None', 'confidence': 0.95},
            {'seed_index': 2, 'seed_name': 'Maize', 'condition': 'Defective / Damaged', 'defect_type': 'Broken', 'confidence': 0.91}
        ]
    }
    
    pdf_out = os.path.join(RESULTS_DIR, 'test_report.pdf')
    res_path = generate_pdf_report(mock_data, pdf_out)
    
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000
