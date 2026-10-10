"""
SeedSense AI - Dataset Summary Report
Generated from real dataset acquisition & organization.
"""

DATASET_REPORT = {
    "total_unified_images": 4250,
    "supported_seed_types": [
        {
            "seed_type": "Rice",
            "status": "AVAILABLE",
            "total_images": 1500,
            "train_count": 1040,
            "val_count": 227,
            "test_count": 233,
            "quality_labels": [
                "Healthy",
                "Defective / Damaged"
            ],
            "defect_labels": [
                "None"
            ]
        },
        {
            "seed_type": "Wheat",
            "status": "AVAILABLE",
            "total_images": 223,
            "train_count": 168,
            "val_count": 28,
            "test_count": 27,
            "quality_labels": [
                "Healthy",
                "Defective / Damaged"
            ],
            "defect_labels": [
                "Broken",
                "Discolored",
                "Skin-damaged"
            ]
        },
        {
            "seed_type": "Maize",
            "status": "AVAILABLE",
            "total_images": 1527,
            "train_count": 1049,
            "val_count": 248,
            "test_count": 230,
            "quality_labels": [
                "Healthy",
                "Defective / Damaged"
            ],
            "defect_labels": [
                "Broken",
                "Skin-damaged",
                "Spotted"
            ]
        },
        {
            "seed_type": "Soybean",
            "status": "AVAILABLE",
            "total_images": 1000,
            "train_count": 718,
            "val_count": 134,
            "test_count": 148,
            "quality_labels": [
                "Healthy",
                "Defective / Damaged"
            ],
            "defect_labels": [
                "Broken",
                "Discolored"
            ]
        }
    ],
    "unsupported_seed_types": [
        {
            "seed_type": "Groundnut / Peanut",
            "status": "INSUFFICIENT REAL DATA",
            "reason": "No suitable public dataset with reliable quality/defect annotations downloaded.",
            "images": 0
        },
        {
            "seed_type": "Chickpea",
            "status": "INSUFFICIENT REAL DATA",
            "reason": "No suitable public dataset with reliable quality/defect annotations downloaded.",
            "images": 0
        },
        {
            "seed_type": "Green Gram / Mung Bean",
            "status": "INSUFFICIENT REAL DATA",
            "reason": "No suitable public dataset with reliable quality/defect annotations downloaded.",
            "images": 0
        },
        {
            "seed_type": "Black Gram / Urad Bean",
            "status": "INSUFFICIENT REAL DATA",
            "reason": "No suitable public dataset with reliable quality/defect annotations downloaded.",
            "images": 0
        },
        {
            "seed_type": "Lentil",
            "status": "INSUFFICIENT REAL DATA",
            "reason": "No suitable public dataset with reliable quality/defect annotations downloaded.",
            "images": 0
        },
        {
            "seed_type": "Sunflower",
            "status": "INSUFFICIENT REAL DATA",
            "reason": "No suitable public dataset with reliable quality/defect annotations downloaded.",
            "images": 0
        }
    ]
}

def print_dataset_report():
    print("=" * 60)
    print("           SEEDSENSE AI - DATASET REPORT            ")
    print("=" * 60)
    print(f"Total Unified Seed Images: {DATASET_REPORT['total_unified_images']}")
    print("-" * 60)
    print("SUPPORTED SEED TYPES:")
    for s in DATASET_REPORT['supported_seed_types']:
        print(f"  - {s['seed_type']} (Status: {s['status']})")
        print(f"    Images: {s['total_images']} (Train: {s['train_count']}, Val: {s['val_count']}, Test: {s['test_count']})")
        print(f"    Quality Labels: {', '.join(s['quality_labels'])}")
        print(f"    Defect Labels: {', '.join(s['defect_labels'])}")
    print("-" * 60)
    print("UNSUPPORTED REQUESTED SEED TYPES:")
    for u in DATASET_REPORT['unsupported_seed_types']:
        print(f"  - {u['seed_type']}: {u['status']}")
    print("=" * 60)

if __name__ == '__main__':
    print_dataset_report()
