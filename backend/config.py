"""
SeedSense AI - Configuration Settings
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
RESULTS_DIR = os.path.join(STATIC_DIR, 'results')
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'seedsense.db')

# Ensure directory creation
for d in [MODELS_DIR, UPLOADS_DIR, RESULTS_DIR, os.path.dirname(DATABASE_PATH)]:
    os.makedirs(d, exist_ok=True)

# Image processing settings
IMAGE_SIZE = (128, 128)

# AI / ML Thresholds
UNKNOWN_CONFIDENCE_THRESHOLD = 0.55

# Classes
SUPPORTED_SPECIES = ['Rice', 'Wheat', 'Maize', 'Soybean']
QUALITY_CLASSES = ['Healthy', 'Defective / Damaged']
DEFECT_CLASSES = ['None', 'Broken', 'Skin-damaged', 'Spotted', 'Discolored']

# Model Paths
SPECIES_MODEL_PATH = os.path.join(MODELS_DIR, 'seed_species_model.pth')
QUALITY_MODEL_PATH = os.path.join(MODELS_DIR, 'seed_quality_model.pth')
DEFECT_MODEL_PATH = os.path.join(MODELS_DIR, 'seed_defect_model.pth')
ML_MODELS_PATH = os.path.join(MODELS_DIR, 'seed_ml_models.joblib')

# Flask Configuration
SECRET_KEY = 'seedsense_ai_secret_key_2026'
