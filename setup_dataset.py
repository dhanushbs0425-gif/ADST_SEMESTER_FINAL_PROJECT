"""
SeedSense AI - Real Dataset Acquisition & Unified Dataset Organization
Downloads, processes, crops, and organizes real public seed image datasets:
1. Rice Seed Dataset (nateraw/rice-image-dataset on Hugging Face - 75,000 real images)
2. Wheat Seed Dataset (Laksh-Singhaniya/SeedVision on GitHub - 1,000 real annotated wheat seed images)
3. Maize/Corn Seed Dataset (AdrianaCasadei/CornGrain on Hugging Face - annotated YOLO bounding box images)
4. Soybean Seed Dataset (lisawen/soybean_dataset on Hugging Face - annotated multi-seed quality segmentation images)

Generates dataset structure, splits (train, val, test), and creates dataset_report.json / dataset_report.py.
"""

import os
import json
import zipfile
import urllib.request
import cv2
import numpy as np
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Subfolders
RAW_DIR = os.path.join(DATASET_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATASET_DIR, 'processed')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# Ensure directories exist
for d in [DATA_DIR, DATASET_DIR, RAW_DIR, PROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# Datasets URLs
DATASET_URLS = {
    'rice': 'https://huggingface.co/datasets/nateraw/rice-image-dataset/resolve/main/rice-image-dataset.zip',
    'corn_train': 'https://huggingface.co/datasets/AdrianaCasadei/CornGrain/resolve/main/train.zip',
    'corn_val': 'https://huggingface.co/datasets/AdrianaCasadei/CornGrain/resolve/main/valid.zip',
    'soybean_train': 'https://huggingface.co/datasets/lisawen/soybean_dataset/resolve/main/train.zip',
    'soybean_val': 'https://huggingface.co/datasets/lisawen/soybean_dataset/resolve/main/valid.zip'
}

# Crop target dimensions
CROP_SIZE = (128, 128)

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[CACHE] Using existing file: {target_path}")
        return
    print(f"[DOWNLOAD] Downloading {url} -> {target_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as f:
        length = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 1024
        while True:
            buffer = resp.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            f.write(buffer)
            if length > 0:
                percent = downloaded / length * 100
                print(f"  Downloaded {downloaded}/{length} bytes ({percent:.1f}%)", end='\r')
    print(f"\n[SUCCESS] Download completed: {target_path}")

def process_rice_dataset(sample_per_variety=300):
    print("\n--- Processing Rice Seed Dataset ---")
    zip_path = os.path.join(DATA_DIR, 'rice-image-dataset.zip')
    download_file(DATASET_URLS['rice'], zip_path)
    
    extracted_crops = []
    varieties = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    with zipfile.ZipFile(zip_path, 'r') as z:
        for var in varieties:
            var_files = [f for f in z.namelist() if f'Rice_Image_Dataset/{var}/' in f and f.endswith('.jpg')]
            random.seed(42)
            selected = random.sample(var_files, min(sample_per_variety, len(var_files)))
            print(f"  Rice variety '{var}': extracting {len(selected)} samples...")
            for idx, file_path in enumerate(selected):
                img_bytes = z.read(file_path)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    img_resized = cv2.resize(img, CROP_SIZE)
                    crop_info = {
                        'species': 'Rice',
                        'condition': 'Healthy',
                        'defect': 'None',
                        'image': img_resized
                    }
                    extracted_crops.append(crop_info)
    print(f"  Total Rice seed crops processed: {len(extracted_crops)}")
    return extracted_crops

def process_wheat_dataset(max_images=200):
    print("\n--- Processing Wheat Seed Dataset (Laksh-Singhaniya/SeedVision) ---")
    extracted_crops = []
    
    # Fetch list of Wheat images from GitHub API
    url = 'https://api.github.com/repos/Laksh-Singhaniya/SeedVision/contents/train/images'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            items = json.loads(resp.read().decode())
            random.seed(42)
            selected_items = random.sample(items, min(max_images, len(items)))
            print(f"  Fetching {len(selected_items)} Wheat seed images from GitHub...")
            
            # Class map based on SeedVision data.yaml
            # 0: black_point, 1: crack, 2: damage, 3: healthy, 4: healthy-50-40, 5: healthy-70-60, 
            # 6: hole, 7: insectdamage, 8: molddamage, 9: objects, 10: shriveledseed, 11: staindamage
            class_map = {
                0: ('Defective / Damaged', 'Spotted'),
                1: ('Defective / Damaged', 'Skin-damaged'),
                2: ('Defective / Damaged', 'Broken'),
                3: ('Healthy', 'None'),
                4: ('Healthy', 'None'),
                5: ('Healthy', 'None'),
                6: ('Defective / Damaged', 'Broken'),
                7: ('Defective / Damaged', 'Discolored'),
                8: ('Defective / Damaged', 'Discolored'),
                10: ('Defective / Damaged', 'Broken'),
                11: ('Defective / Damaged', 'Discolored')
            }
            
            for item in selected_items:
                img_url = item['download_url']
                lbl_url = img_url.replace('/train/images/', '/train/labels/').replace('.jpeg', '.txt').replace('.jpg', '.txt')
                
                try:
                    with urllib.request.urlopen(img_url) as img_resp:
                        img_data = img_resp.read()
                        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                        
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    
                    try:
                        with urllib.request.urlopen(lbl_url) as lbl_resp:
                            lbl_lines = lbl_resp.read().decode('utf-8').strip().split('\n')
                    except Exception:
                        lbl_lines = []
                        
                    if not lbl_lines or not lbl_lines[0]:
                        # Default whole image crop
                        crop_resized = cv2.resize(img, CROP_SIZE)
                        extracted_crops.append({
                            'species': 'Wheat',
                            'condition': 'Healthy',
                            'defect': 'None',
                            'image': crop_resized
                        })
                        continue
                        
                    for line in lbl_lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        if cls_id not in class_map:
                            continue
                            
                        # If segment polygon coordinates present
                        if len(parts) > 5:
                            coords = [float(x) for x in parts[1:]]
                            xs = coords[0::2]
                            ys = coords[1::2]
                            x1 = int(max(0, min(xs) * w))
                            y1 = int(max(0, min(ys) * h))
                            x2 = int(min(w, max(xs) * w))
                            y2 = int(min(h, max(ys) * h))
                        else:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = int(max(0, (cx - bw / 2) * w))
                            y1 = int(max(0, (cy - bh / 2) * h))
                            x2 = int(min(w, (cx + bw / 2) * w))
                            y2 = int(min(h, (cy + bh / 2) * h))
                            
                        if (x2 - x1) < 15 or (y2 - y1) < 15:
                            continue
                            
                        crop = img[y1:y2, x1:x2]
                        crop_resized = cv2.resize(crop, CROP_SIZE)
                        cond, defect = class_map[cls_id]
                        
                        extracted_crops.append({
                            'species': 'Wheat',
                            'condition': cond,
                            'defect': defect,
                            'image': crop_resized
                        })
                except Exception as e:
                    pass
    except Exception as e:
        print("  Warning: could not fetch Wheat dataset from GitHub:", e)
        
    print(f"  Total Wheat seed crops extracted: {len(extracted_crops)}")
    return extracted_crops

def process_corn_dataset(max_crops_per_class=400):
    print("\n--- Processing Maize/Corn Seed Dataset ---")
    corn_train_path = os.path.join(DATA_DIR, 'corn_train.zip')
    corn_val_path = os.path.join(DATA_DIR, 'corn_val.zip')
    
    download_file(DATASET_URLS['corn_train'], corn_train_path)
    download_file(DATASET_URLS['corn_val'], corn_val_path)
    
    class_map = {
        0: ('Defective / Damaged', 'Broken'),
        1: ('Healthy', 'None'),
        2: ('Healthy', 'None'),
        3: ('Defective / Damaged', 'Skin-damaged'),
        4: ('Defective / Damaged', 'Spotted')
    }
    
    extracted_crops = []
    class_counts = {k: 0 for k in class_map.keys()}
    
    for zip_path in [corn_train_path, corn_val_path]:
        with zipfile.ZipFile(zip_path, 'r') as z:
            txt_files = [f for f in z.namelist() if f.endswith('.txt') and not f.startswith('__MACOSX') and 'labels/' in f]
            for txt_file in txt_files:
                img_file = txt_file.replace('labels/', 'images/').replace('.txt', '.jpg')
                if img_file not in z.namelist():
                    img_file = img_file.replace('.jpg', '.png')
                if img_file not in z.namelist():
                    continue
                
                txt_content = z.read(txt_file).decode('utf-8').strip().split('\n')
                if not txt_content or not txt_content[0]:
                    continue
                
                img_bytes = z.read(img_file)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                for line in txt_content:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id not in class_map:
                        continue
                    
                    if class_counts[cls_id] >= max_crops_per_class:
                        continue
                    
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int(max(0, (cx - bw / 2) * w))
                    y1 = int(max(0, (cy - bh / 2) * h))
                    x2 = int(min(w, (cx + bw / 2) * w))
                    y2 = int(min(h, (cy + bh / 2) * h))
                    
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    
                    crop = img[y1:y2, x1:x2]
                    crop_resized = cv2.resize(crop, CROP_SIZE)
                    
                    cond, defect = class_map[cls_id]
                    extracted_crops.append({
                        'species': 'Maize',
                        'condition': cond,
                        'defect': defect,
                        'image': crop_resized
                    })
                    class_counts[cls_id] += 1
                    
    print(f"  Total Maize seed crops extracted: {len(extracted_crops)}")
    return extracted_crops

def process_soybean_dataset(max_crops=1000):
    print("\n--- Processing Soybean Seed Dataset ---")
    soy_train_path = os.path.join(DATA_DIR, 'soybean_train.zip')
    soy_val_path = os.path.join(DATA_DIR, 'soybean_valid.zip')
    
    download_file(DATASET_URLS['soybean_train'], soy_train_path)
    download_file(DATASET_URLS['soybean_val'], soy_val_path)
    
    extracted_crops = []
    
    for zip_path in [soy_train_path, soy_val_path]:
        with zipfile.ZipFile(zip_path, 'r') as z:
            orig_files = [f for f in z.namelist() if f.endswith('_original.jpg') and not f.startswith('__MACOSX')]
            for orig_file in orig_files:
                if len(extracted_crops) >= max_crops:
                    break
                seg_file = orig_file.replace('_original.jpg', '_segmentation.jpg')
                if seg_file not in z.namelist():
                    continue
                
                orig_bytes = z.read(orig_file)
                seg_bytes = z.read(seg_file)
                
                orig_img = cv2.imdecode(np.frombuffer(orig_bytes, np.uint8), cv2.IMREAD_COLOR)
                seg_img = cv2.imdecode(np.frombuffer(seg_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if orig_img is None or seg_img is None:
                    continue
                
                gray_seg = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_seg, 20, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 300 or area > 50000:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(c)
                    crop_orig = orig_img[y:y+h, x:x+w]
                    crop_seg = seg_img[y:y+h, x:x+w]
                    
                    if crop_orig.shape[0] < 15 or crop_orig.shape[1] < 15:
                        continue
                    
                    mean_color_bgr = cv2.mean(crop_seg)[:3]
                    b, g, r = mean_color_bgr
                    
                    if g > r + 30 and g > b + 30:
                        cond = 'Healthy'
                        defect = 'None'
                    elif r > g + 20 and r > b + 20:
                        cond = 'Defective / Damaged'
                        defect = 'Broken'
                    elif r > b + 20 and g > b + 20:
                        cond = 'Defective / Damaged'
                        defect = 'Discolored'
                    else:
                        cond = 'Healthy'
                        defect = 'None'
                        
                    crop_resized = cv2.resize(crop_orig, CROP_SIZE)
                    extracted_crops.append({
                        'species': 'Soybean',
                        'condition': cond,
                        'defect': defect,
                        'image': crop_resized
                    })
                    
                    if len(extracted_crops) >= max_crops:
                        break
                        
    print(f"  Total Soybean seed crops extracted: {len(extracted_crops)}")
    return extracted_crops

def organize_and_save(all_crops):
    print("\n--- Organizing & Splitting Unified Dataset ---")
    random.seed(42)
    random.shuffle(all_crops)
    
    total = len(all_crops)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)
    
    train_crops = all_crops[:train_end]
    val_crops = all_crops[train_end:val_end]
    test_crops = all_crops[val_end:]
    
    splits = {
        'train': (TRAIN_DIR, train_crops),
        'val': (VAL_DIR, val_crops),
        'test': (TEST_DIR, test_crops)
    }
    
    stats = {}
    
    for split_name, (split_dir, crops) in splits.items():
        count = 0
        for item in crops:
            species = item['species']
            cond = 'healthy' if item['condition'] == 'Healthy' else 'defective'
            defect = item['defect'].lower().replace(' ', '_').replace('/', '')
            
            target_dir = os.path.join(split_dir, species.lower(), cond)
            if cond == 'defective':
                target_dir = os.path.join(target_dir, defect)
            os.makedirs(target_dir, exist_ok=True)
            
            filename = f"{species.lower()}_{cond}_{defect}_{count:05d}.jpg"
            filepath = os.path.join(target_dir, filename)
            cv2.imwrite(filepath, item['image'])
            count += 1
            
            key = f"{species}|{item['condition']}|{item['defect']}"
            if key not in stats:
                stats[key] = {'train': 0, 'val': 0, 'test': 0}
            stats[key][split_name] += 1
            
    print(f"[COMPLETE] Saved {total} organized crops into train/val/test splits.")
    return stats, total

def generate_reports(stats, total_crops):
    print("\n--- Generating Dataset Report ---")
    
    supported_species = ['Rice', 'Wheat', 'Maize', 'Soybean']
    unsupported_requested = [
        'Groundnut / Peanut', 'Chickpea', 
        'Green Gram / Mung Bean', 'Black Gram / Urad Bean', 
        'Lentil', 'Sunflower'
    ]
    
    species_summary = {}
    for item_key, counts in stats.items():
        species, cond, defect = item_key.split('|')
        if species not in species_summary:
            species_summary[species] = {
                'total_images': 0,
                'healthy_count': 0,
                'defective_count': 0,
                'defects_found': set(),
                'train': 0, 'val': 0, 'test': 0
            }
        tot = counts['train'] + counts['val'] + counts['test']
        species_summary[species]['total_images'] += tot
        species_summary[species]['train'] += counts['train']
        species_summary[species]['val'] += counts['val']
        species_summary[species]['test'] += counts['test']
        
        if cond == 'Healthy':
            species_summary[species]['healthy_count'] += tot
        else:
            species_summary[species]['defective_count'] += tot
            species_summary[species]['defects_found'].add(defect)
            
    report_data = {
        'total_unified_images': total_crops,
        'supported_seed_types': [],
        'unsupported_seed_types': []
    }
    
    for s in supported_species:
        info = species_summary.get(s, {})
        defects = sorted(list(info.get('defects_found', [])))
        report_data['supported_seed_types'].append({
            'seed_type': s,
            'status': 'AVAILABLE',
            'total_images': info.get('total_images', 0),
            'train_count': info.get('train', 0),
            'val_count': info.get('val', 0),
            'test_count': info.get('test', 0),
            'quality_labels': ['Healthy', 'Defective / Damaged'],
            'defect_labels': defects if defects else ['None']
        })
        
    for u in unsupported_requested:
        report_data['unsupported_seed_types'].append({
            'seed_type': u,
            'status': 'INSUFFICIENT REAL DATA',
            'reason': 'No suitable public dataset with reliable quality/defect annotations downloaded.',
            'images': 0
        })
        
    report_json_path = os.path.join(DATASET_DIR, 'dataset_report.json')
    with open(report_json_path, 'w') as f:
        json.dump(report_data, f, indent=4)
        
    print(f"[REPORT] Saved JSON report at: {report_json_path}")
    
    py_report_path = os.path.join(BASE_DIR, 'dataset_report.py')
    py_content = f'''"""
SeedSense AI - Dataset Summary Report
Generated from real dataset acquisition & organization.
"""

DATASET_REPORT = {json.dumps(report_data, indent=4)}

def print_dataset_report():
    print("=" * 60)
    print("           SEEDSENSE AI - DATASET REPORT            ")
    print("=" * 60)
    print(f"Total Unified Seed Images: {{DATASET_REPORT['total_unified_images']}}")
    print("-" * 60)
    print("SUPPORTED SEED TYPES:")
    for s in DATASET_REPORT['supported_seed_types']:
        print(f"  - {{s['seed_type']}} (Status: {{s['status']}})")
        print(f"    Images: {{s['total_images']}} (Train: {{s['train_count']}}, Val: {{s['val_count']}}, Test: {{s['test_count']}})")
        print(f"    Quality Labels: {{', '.join(s['quality_labels'])}}")
        print(f"    Defect Labels: {{', '.join(s['defect_labels'])}}")
    print("-" * 60)
    print("UNSUPPORTED REQUESTED SEED TYPES:")
    for u in DATASET_REPORT['unsupported_seed_types']:
        print(f"  - {{u['seed_type']}}: {{u['status']}}")
    print("=" * 60)

if __name__ == '__main__':
    print_dataset_report()
'''
    with open(py_report_path, 'w') as f:
        f.write(py_content)
    print(f"[REPORT] Saved Python report module at: {py_report_path}")

def main():
    print("==================================================")
    print("       SEEDSENSE AI DATASET ACQUISITION          ")
    print("==================================================")
    
    rice_crops = process_rice_dataset(sample_per_variety=300)
    wheat_crops = process_wheat_dataset(max_images=200)
    corn_crops = process_corn_dataset(max_crops_per_class=350)
    soybean_crops = process_soybean_dataset(max_crops=1000)
    
    all_crops = rice_crops + wheat_crops + corn_crops + soybean_crops
    print(f"\nTotal extracted real seed crops: {len(all_crops)}")
    
    stats, total_saved = organize_and_save(all_crops)
    generate_reports(stats, total_saved)
    print("\nDataset setup successfully finished!")

if __name__ == '__main__':
    main()
