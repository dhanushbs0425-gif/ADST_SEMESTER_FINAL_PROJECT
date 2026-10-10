"""
SeedSense AI - Tray / Batch Image Analyzer & Visual Annotation Engine
Segment seeds -> Predict per seed -> Generate annotated image -> Calculate statistics
"""

import os
import cv2
import time
import numpy as np

from backend.config import RESULTS_DIR
from backend.segmentation import segment_seeds
from backend.prediction import predictor

def analyze_seed_image(image_path, model_type='dl'):
    """
    Main entry point for Single & Batch seed image analysis.
    Performs segmentation, quality inspection, defect detection, visual annotation, and statistics calculation.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image file: {image_path}")
        
    # Segment individual seeds
    seed_crops, annotated_img = segment_seeds(img_bgr)
    
    predictions = []
    
    total_seeds = len(seed_crops)
    healthy_count = 0
    defective_count = 0
    
    species_counts = {}
    species_healthy = {}
    species_defective = {}
    defect_counts = {}
    
    for item in seed_crops:
        crop_bgr = item['crop']
        x, y, w, h = item['bbox']
        
        # Run prediction
        res = predictor.predict_crop(crop_bgr, model_type=model_type)
        predictions.append(res)
        
        sp = res['seed_name']
        cond = res['condition']
        defect = res['defect_type']
        conf = res['confidence']
        
        # Track statistics
        species_counts[sp] = species_counts.get(sp, 0) + 1
        if sp not in species_healthy:
            species_healthy[sp] = 0
            species_defective[sp] = 0
            
        if 'Healthy' in cond:
            healthy_count += 1
            species_healthy[sp] += 1
        else:
            defective_count += 1
            species_defective[sp] += 1
            if defect != 'None' and defect != 'Unknown':
                defect_counts[defect] = defect_counts.get(defect, 0) + 1
                
        # Draw Visual Annotations
        color = (0, 200, 0) if 'Healthy' in cond else (0, 0, 220) # Green for Healthy, Red for Defective
        
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
        
        # Annotation text lines focusing on Quality & Defect
        cond_short = cond.split('/')[0].strip()
        label1 = f"{cond_short} ({int(conf*100)}%)"
        label2 = f"Defect: {defect}" if defect != 'None' else "Intact / Healthy"
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        
        text_y = max(y - 8, 15)
        cv2.putText(annotated_img, label1, (x, text_y), font, scale, color, thick, cv2.LINE_AA)
        cv2.putText(annotated_img, label2, (x, min(y + h + 15, annotated_img.shape[0] - 5)), font, scale - 0.05, color, thick, cv2.LINE_AA)
        
    # Overall Percentages
    healthy_pct = round((healthy_count / total_seeds * 100), 2) if total_seeds > 0 else 0.0
    defective_pct = round((defective_count / total_seeds * 100), 2) if total_seeds > 0 else 0.0
    
    # Per-species quality breakdown
    per_species_quality = {}
    for sp_name, total_sp in species_counts.items():
        h_cnt = species_healthy.get(sp_name, 0)
        d_cnt = species_defective.get(sp_name, 0)
        h_pct = round((h_cnt / total_sp * 100), 2) if total_sp > 0 else 0.0
        per_species_quality[sp_name] = {
            'total': total_sp,
            'healthy': h_cnt,
            'defective': d_cnt,
            'healthy_percentage': h_pct
        }
        
    # Save Annotated Image
    timestamp = int(time.time() * 1000)
    annotated_filename = f"result_{timestamp}.jpg"
    annotated_filepath = os.path.join(RESULTS_DIR, annotated_filename)
    cv2.imwrite(annotated_filepath, annotated_img)
    
    # Overwrite standard static/results/annotated_result.jpg
    standard_filepath = os.path.join(RESULTS_DIR, "annotated_result.jpg")
    cv2.imwrite(standard_filepath, annotated_img)
    
    primary_res = predictions[0] if total_seeds == 1 else {
        "seed_name": list(species_counts.keys())[0] if species_counts else "Seed",
        "condition": "Healthy" if healthy_count >= defective_count else "Defective / Damaged",
        "defect_type": list(defect_counts.keys())[0] if defect_counts else "None",
        "confidence": round(float(np.mean([p['confidence'] for p in predictions])), 4) if predictions else 0.0
    }
    
    analysis_result = {
        'timestamp': timestamp,
        'original_image_path': image_path,
        'annotated_image_filename': annotated_filename,
        'annotated_image_path': annotated_filepath,
        'primary_seed_name': primary_res['seed_name'],
        'primary_condition': primary_res['condition'],
        'primary_defect_type': primary_res['defect_type'],
        'primary_confidence': primary_res['confidence'],
        'total_seeds': total_seeds,
        'healthy_count': healthy_count,
        'defective_count': defective_count,
        'healthy_percentage': healthy_pct,
        'defective_percentage': defective_pct,
        'species_distribution': species_counts,
        'defect_distribution': defect_counts,
        'per_species_quality': per_species_quality,
        'predictions': predictions,
        'model_used': model_type.upper()
    }
    
    return analysis_result
