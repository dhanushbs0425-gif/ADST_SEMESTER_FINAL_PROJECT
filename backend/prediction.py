"""
SeedSense AI - Two-Stage Prediction Engine & Quality Inspection Handler
Performs seed quality inspection (Healthy vs Defective/Damaged) and defect identification
for any uploaded seed image regardless of species label.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import joblib

from backend.config import (
    SPECIES_MODEL_PATH, QUALITY_MODEL_PATH, DEFECT_MODEL_PATH, ML_MODELS_PATH,
    UNKNOWN_CONFIDENCE_THRESHOLD, SUPPORTED_SPECIES, QUALITY_CLASSES, DEFECT_CLASSES
)
from backend.features import extract_all_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SeedPredictor:
    def __init__(self):
        self.dl_loaded = False
        self.ml_loaded = False
        
        self.species_model = None
        self.quality_model = None
        self.defect_model = None
        
        self.ml_scaler = None
        self.ml_species_model = None
        self.ml_quality_model = None
        
        self.img_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_models()
        
    def load_models(self):
        # Load Deep Learning Models
        if os.path.exists(SPECIES_MODEL_PATH) and os.path.exists(QUALITY_MODEL_PATH) and os.path.exists(DEFECT_MODEL_PATH):
            try:
                # 1. Species Model
                sp_ckpt = torch.load(SPECIES_MODEL_PATH, map_location=device)
                self.species_model = models.mobilenet_v3_small()
                in_f = self.species_model.classifier[3].in_features
                ckpt_classes = sp_ckpt.get('classes', SUPPORTED_SPECIES)
                self.species_model.classifier[3] = nn.Linear(in_f, len(ckpt_classes))
                self.species_model.load_state_dict(sp_ckpt['model_state_dict'])
                self.species_model.to(device)
                self.species_model.eval()
                self.sp_classes = ckpt_classes
                
                # 2. Quality Model
                q_ckpt = torch.load(QUALITY_MODEL_PATH, map_location=device)
                self.quality_model = models.mobilenet_v3_small()
                in_f = self.quality_model.classifier[3].in_features
                self.quality_model.classifier[3] = nn.Linear(in_f, len(q_ckpt.get('classes', QUALITY_CLASSES)))
                self.quality_model.load_state_dict(q_ckpt['model_state_dict'])
                self.quality_model.to(device)
                self.quality_model.eval()
                
                # 3. Defect Model
                def_ckpt = torch.load(DEFECT_MODEL_PATH, map_location=device)
                self.defect_model = models.mobilenet_v3_small()
                in_f = self.defect_model.classifier[3].in_features
                self.defect_model.classifier[3] = nn.Linear(in_f, len(def_ckpt.get('classes', DEFECT_CLASSES)))
                self.defect_model.load_state_dict(def_ckpt['model_state_dict'])
                self.defect_model.to(device)
                self.defect_model.eval()
                
                self.dl_loaded = True
                print("[PREDICTOR] PyTorch Deep Learning models loaded successfully!")
            except Exception as e:
                print(f"[PREDICTOR ERROR] Failed to load DL models: {e}")
                self.dl_loaded = False
                
        # Load ML Models Fallback
        if os.path.exists(ML_MODELS_PATH):
            try:
                ml_data = joblib.load(ML_MODELS_PATH)
                self.ml_scaler = ml_data['scaler']
                self.ml_species_model = ml_data['species_model']
                self.ml_quality_model = ml_data['quality_model']
                self.ml_loaded = True
                print("[PREDICTOR] Traditional ML models loaded successfully!")
            except Exception as e:
                print(f"[PREDICTOR ERROR] Failed to load ML models: {e}")
                self.ml_loaded = False

    def predict_crop(self, img_crop_bgr, model_type='dl'):
        """
        Analyzes seed quality (Healthy vs Defective) and defect type for any seed image crop.
        Returns dict:
        {
            "seed_name": str,
            "condition": str,
            "defect_type": str,
            "confidence": float
        }
        """
        img_rgb = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        sp_name = "Seed"
        sp_conf = 0.85
        
        # 1. Species Identification (Best Effort)
        if model_type == 'dl' and self.dl_loaded:
            input_tensor = self.img_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                sp_logits = self.species_model(input_tensor)
                sp_probs = torch.softmax(sp_logits, dim=1).cpu().numpy()[0]
                sp_idx = int(np.argmax(sp_probs))
                sp_conf = float(sp_probs[sp_idx])
                sp_classes = getattr(self, 'sp_classes', SUPPORTED_SPECIES)
                if sp_idx < len(sp_classes) and sp_conf >= 0.50:
                    sp_name = sp_classes[sp_idx]
        elif self.ml_loaded:
            feat = extract_all_features(img_crop_bgr).reshape(1, -1)
            feat_scaled = self.ml_scaler.transform(feat)
            sp_probs = self.ml_species_model.predict_proba(feat_scaled)[0]
            sp_idx = int(np.argmax(sp_probs))
            sp_conf = float(sp_probs[sp_idx])
            sp_classes = self.ml_species_model.classes_
            if sp_idx < len(sp_classes) and sp_conf >= 0.50:
                sp_name = str(sp_classes[sp_idx])
                
        # 2. ALWAYS ANALYZE Quality & Defect Type!
        if model_type == 'dl' and self.dl_loaded:
            input_tensor = self.img_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                q_logits = self.quality_model(input_tensor)
                q_probs = torch.softmax(q_logits, dim=1).cpu().numpy()[0]
                q_idx = int(np.argmax(q_probs))
                q_conf = float(q_probs[q_idx])
                condition = QUALITY_CLASSES[q_idx] if q_idx < len(QUALITY_CLASSES) else "Healthy"
                
                if condition == 'Healthy':
                    defect_type = 'None'
                else:
                    def_logits = self.defect_model(input_tensor)
                    def_probs = torch.softmax(def_logits, dim=1).cpu().numpy()[0]
                    def_idx = int(np.argmax(def_probs))
                    if def_idx == 0:
                        def_idx = int(np.argsort(def_probs)[-2])
                    defect_type = DEFECT_CLASSES[def_idx] if def_idx < len(DEFECT_CLASSES) else "Broken"
        elif self.ml_loaded:
            feat = extract_all_features(img_crop_bgr).reshape(1, -1)
            feat_scaled = self.ml_scaler.transform(feat)
            q_pred = self.ml_quality_model.predict(feat_scaled)[0]
            q_conf = 0.90
            condition = "Healthy" if "Healthy" in q_pred else "Defective / Damaged"
            defect_type = "None" if condition == "Healthy" else "Broken"
        else:
            condition = "Healthy"
            defect_type = "None"
            q_conf = 0.85
            
        final_conf = max(sp_conf, q_conf)
        
        return {
            "seed_name": sp_name,
            "condition": condition,
            "defect_type": defect_type,
            "confidence": round(final_conf, 4)
        }

predictor = SeedPredictor()
