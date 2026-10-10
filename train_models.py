"""
SeedSense AI - Comprehensive Model Training Pipeline
Trains Traditional ML Models (SVM, Random Forest, KNN, Logistic Regression)
and Deep Learning Models (MobileNetV3 for Species, Quality, and Defect classification).
"""

import time
from backend.ml_engine import train_and_evaluate_ml
from backend.species_model import train_species_model
from backend.quality_model import train_quality_model
from backend.defect_model import train_defect_model

def run_all_training():
    print("==================================================")
    print("   STARTING SEEDSENSE AI FULL MODEL TRAINING      ")
    print("==================================================")
    start_time = time.time()
    
    # 1. Train traditional ML models
    ml_results = train_and_evaluate_ml()
    
    # 2. Train Deep Learning Stage 1: Species Model
    species_results = train_species_model(epochs=4, batch_size=32)
    
    # 3. Train Deep Learning Stage 2: Quality Model
    quality_results = train_quality_model(epochs=4, batch_size=32)
    
    # 4. Train Deep Learning Stage 2: Defect Model
    defect_results = train_defect_model(epochs=4, batch_size=32)
    
    elapsed = time.time() - start_time
    print("==================================================")
    print(f"[FINISHED] All AI/ML models trained in {elapsed:.2f} seconds!")
    print("==================================================")

if __name__ == '__main__':
    run_all_training()
