## Automated Seed Quality Identification and Defect Classification Using Deep Learning and Computer Vision
SeedSense AI is a complete, full-stack AI/ML, Computer Vision, Deep Learning, and Data Science project that provides automated seed species identification, health quality assessment (Healthy vs. Defective/Damaged), and detailed defect categorization.

---

## 🌟 Key Features

1. **Two-Stage AI Classification Architecture**:
   - **Stage 1**: Seed Species Identification (Rice, Maize, Soybean)
   - **Stage 2**: Seed Quality Assessment (Healthy vs. Defective/Damaged) & Defect Type Identification (Broken, Skin-damaged, Spotted, Discolored)
2. **Computer Vision & Watershed Segmentation**:
   - Automated separation of multi-seed tray images into individual crops using OpenCV Distance Transform & Watershed Segmentation.
3. **Traditional ML & Deep Learning Support**:
   - **Deep Learning**: PyTorch Transfer Learning using MobileNetV3 Small.
   - **Machine Learning**: Color (RGB/HSV), Shape (Solidity, Extent, Aspect Ratio), and Texture (GLCM Contrast, Homogeneity, Energy, Correlation) + SVM / Random Forest / KNN / Logistic Regression.
4. **Unknown / Unsupported Seed Handling**:
   - Rejects non-seed or unsupported objects when confidence < 0.55 threshold without forcing false predictions.
5. **Full Web Application & Dashboard**:
   - Built with Flask, Bootstrap 5, and Chart.js.
6. **SQLite Database & PDF Reports**:
   - Analysis history persistence and downloadable PDF reports generated via ReportLab.

---

## 📁 Project Directory Structure

```
SeedSense-AI/
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── analyzer.py
│   ├── segmentation.py
│   ├── features.py
│   ├── ml_engine.py
│   ├── species_model.py
│   ├── quality_model.py
│   ├── defect_model.py
│   ├── prediction.py
│   ├── database.py
│   └── report.py
├── dataset/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
├── data/
│   └── seedsense.db
├── models/
│   ├── seed_species_model.pth
│   ├── seed_quality_model.pth
│   ├── seed_defect_model.pth
│   └── seed_ml_models.joblib
├── uploads/
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── results/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── results.html
│   ├── batch.html
│   ├── history.html
│   ├── performance.html
│   ├── dataset_info.html
│   └── about.html
├── tests/
│   ├── test_dataset.py
│   ├── test_features.py
│   ├── test_segmentation.py
│   ├── test_prediction.py
│   ├── test_database.py
│   ├── test_report.py
│   └── test_app.py
├── setup_dataset.py
├── dataset_report.py
├── train_models.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Commands

```bash
# 1. Setup real datasets
python setup_dataset.py

# 2. Train AI & ML models
python train_models.py

# 3. Execute test suite
pytest

# 4. Start Flask web server
python -m backend.app
```

URL to open web application: `http://localhost:5000`
