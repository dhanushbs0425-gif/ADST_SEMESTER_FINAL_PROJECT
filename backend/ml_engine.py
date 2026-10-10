"""
SeedSense AI - Traditional Machine Learning Engine
Extracts features and trains SVM, Random Forest, KNN, and Logistic Regression models.
Computes REAL accuracy, precision, recall, F1-score, and confusion matrices.
"""

import os
import cv2
import glob
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from backend.config import TRAIN_DIR, TEST_DIR, ML_MODELS_PATH, IMAGE_SIZE
from backend.features import extract_all_features

def load_dataset_features(split_dir):
    """
    Scans dataset directory (train/test), extracts 33-D features, and returns X, y_species, y_quality.
    """
    X = []
    y_species = []
    y_quality = []
    
    print(f"[ML DATA] Loading images from: {split_dir}")
    
    image_paths = glob.glob(os.path.join(split_dir, '**', '*.jpg'), recursive=True)
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_resized = cv2.resize(img, IMAGE_SIZE)
        feat = extract_all_features(img_resized)
        
        # Parse path for labels
        # Path format: split_dir/species/condition/...
        rel_path = os.path.relpath(img_path, split_dir)
        parts = rel_path.split(os.sep)
        
        if len(parts) >= 2:
            species = parts[0].capitalize() # e.g. Rice, Maize, Soybean
            condition = parts[1].capitalize() # e.g. Healthy, Defective
            
            X.append(feat)
            y_species.append(species)
            y_quality.append('Healthy' if 'Healthy' in condition else 'Defective')
            
    return np.array(X), np.array(y_species), np.array(y_quality)

from sklearn.base import clone

def train_and_evaluate_ml():
    """
    Trains traditional ML models and computes REAL evaluation metrics on the TEST dataset.
    """
    print("==================================================")
    print("     TRAINING TRADITIONAL MACHINE LEARNING MODELS ")
    print("==================================================")
    
    X_train, y_sp_train, y_qual_train = load_dataset_features(TRAIN_DIR)
    X_test, y_sp_test, y_qual_test = load_dataset_features(TEST_DIR)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Dataset features empty! Run setup_dataset.py first.")
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Classifiers to evaluate
    classifiers = {
        'SVM': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42)
    }
    
    species_results = {}
    quality_results = {}
    
    trained_models = {
        'scaler': scaler,
        'species_model': None,
        'quality_model': None,
        'best_species_classifier': None,
        'best_quality_classifier': None
    }
    
    best_sp_acc = 0.0
    best_qual_acc = 0.0
    
    print("\n--- 1. SPECIES CLASSIFICATION RESULTS ---")
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_sp_train)
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_sp_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_sp_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_sp_test, y_pred).tolist()
        
        labels = sorted(list(set(y_sp_test)))
        
        species_results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm,
            'labels': labels
        }
        
        print(f"  [{name}] Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")
        
        if acc > best_sp_acc:
            best_sp_acc = acc
            trained_models['species_model'] = clf
            trained_models['best_species_classifier'] = name
            
    print("\n--- 2. QUALITY CLASSIFICATION RESULTS ---")
    for name, clf in classifiers.items():
        # Train separate instance for quality classification
        clf_qual = clone(clf)
        clf_qual.fit(X_train_scaled, y_qual_train)
        y_pred = clf_qual.predict(X_test_scaled)
        
        acc = accuracy_score(y_qual_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_qual_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_qual_test, y_pred).tolist()
        
        labels = sorted(list(set(y_qual_test)))
        
        quality_results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm,
            'labels': labels
        }
        
        print(f"  [{name}] Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")
        
        if acc > best_qual_acc:
            best_qual_acc = acc
            trained_models['quality_model'] = clf_qual
            trained_models['best_quality_classifier'] = name
            
    # Save best models and evaluation metrics to joblib
    ml_data_to_save = {
        'scaler': trained_models['scaler'],
        'species_model': trained_models['species_model'],
        'quality_model': trained_models['quality_model'],
        'best_species_classifier': trained_models['best_species_classifier'],
        'best_quality_classifier': trained_models['best_quality_classifier'],
        'species_results': species_results,
        'quality_results': quality_results
    }
    
    joblib.dump(ml_data_to_save, ML_MODELS_PATH)
    print(f"\n[SAVE] Saved ML models and metrics to: {ML_MODELS_PATH}")
    
    return ml_data_to_save

if __name__ == '__main__':
    train_and_evaluate_ml()
