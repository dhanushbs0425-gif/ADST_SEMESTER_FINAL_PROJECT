"""
Integration Tests for Flask Web Application Routes
"""

import os
import io
import pytest

from backend.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'SeedSense' in rv.data

def test_upload_route_get(client):
    rv = client.get('/upload')
    assert rv.status_code == 200
    assert b'Upload Seed Image' in rv.data

def test_batch_route_get(client):
    rv = client.get('/batch')
    assert rv.status_code == 200
    assert b'Batch Analysis' in rv.data

def test_history_route(client):
    rv = client.get('/history')
    assert rv.status_code == 200
    assert b'History' in rv.data

def test_performance_route(client):
    rv = client.get('/performance')
    assert rv.status_code == 200
    assert b'Performance' in rv.data

def test_dataset_info_route(client):
    rv = client.get('/dataset_info')
    assert rv.status_code == 200
    assert b'Dataset Information' in rv.data

def test_about_route(client):
    rv = client.get('/about')
    assert rv.status_code == 200
    assert b'About' in rv.data
