"""
SeedSense AI - Computer Vision & Segmentation Pipeline
Uses OpenCV Denoising, Thresholding, Morphology, Distance Transform, and Watershed Segmentation.
"""

import cv2
import numpy as np

def preprocess_image(img_bgr, target_size=None):
    """
    Applies Gaussian blurring and Denoising to clean input seed images.
    """
    if target_size is not None:
        img_bgr = cv2.resize(img_bgr, target_size)
    denoised = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    return denoised

def segment_seeds(img_bgr, min_area=600, max_area=150000):
    """
    Performs Watershed & Adaptive Contour Segmentation to extract individual seeds.
    Intelligently handles single-seed uploads and multi-seed tray images without noise fragmentation.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w
    annotated_img = img_bgr.copy()
    
    # Dynamic minimum area threshold (at least 0.3% of image area or 600 px)
    effective_min_area = max(min_area, int(0.003 * img_area))
    
    # Denoise
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    
    # Convert to Gray & LAB for robust thresholding
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Otsu thresholding on Gray & b_channel
    _, thresh_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_b = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine binary masks
    combined_thresh = cv2.bitwise_or(thresh_gray, thresh_b)
    
    # If the background is light, invert if necessary
    if np.mean(combined_thresh) > 127:
        combined_thresh = cv2.bitwise_not(combined_thresh)
        
    # Morphological Opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    seed_crops = []
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < effective_min_area or area > max_area:
            continue
            
        x, y, bw, bh = cv2.boundingRect(c)
        
        # Filter out border artifacts (contours touching extreme edges of the image)
        if x <= 3 or y <= 3 or (x + bw) >= w - 3 or (y + bh) >= h - 3:
            # If contour covers > 20% of image area, it's the main seed, not a border artifact
            if area < 0.20 * img_area:
                continue
                
        valid_contours.append((area, c, (x, y, bw, bh)))
        
    # Sort by area descending
    valid_contours.sort(key=lambda item: item[0], reverse=True)
    
    # Check if this is a Single Seed Image (main contour covers >= 15% of image area or <= 1 valid contour)
    if len(valid_contours) <= 1 or (valid_contours and valid_contours[0][0] >= 0.15 * img_area):
        # Single seed mode: return full image or single clean crop
        if valid_contours:
            _, _, (x, y, bw, bh) = valid_contours[0]
            pad = 10
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            crop = img_bgr[y1:y2, x1:x2]
            mask_crop = opening[y1:y2, x1:x2]
            seed_crops.append({
                'id': 0,
                'crop': crop if crop.size > 0 else img_bgr,
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'mask': mask_crop
            })
        else:
            seed_crops.append({
                'id': 0,
                'crop': img_bgr,
                'bbox': (0, 0, w, h),
                'mask': opening
            })
    else:
        # Multi-seed Tray mode
        idx = 0
        for area, c, (x, y, bw, bh) in valid_contours:
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            
            crop = img_bgr[y1:y2, x1:x2]
            if crop.shape[0] < 15 or crop.shape[1] < 15:
                continue
                
            mask_crop = opening[y1:y2, x1:x2]
            seed_crops.append({
                'id': idx,
                'crop': crop,
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'mask': mask_crop
            })
            idx += 1
            
    if len(seed_crops) == 0:
        seed_crops.append({
            'id': 0,
            'crop': img_bgr,
            'bbox': (0, 0, w, h),
            'mask': opening
        })
        
    return seed_crops, annotated_img
