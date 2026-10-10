"""
SeedSense AI - Stage 2: Seed Defect Type Classification Model (Defect Types)
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from backend.config import TRAIN_DIR, VAL_DIR, TEST_DIR, DEFECT_MODEL_PATH, DEFECT_CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SeedDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.defect_map = {d.lower(): i for i, d in enumerate(DEFECT_CLASSES)}
        
        for p in glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True):
            rel = os.path.relpath(p, root_dir)
            parts = rel.split(os.sep)
            
            target = 0 # Default 'None' for Healthy
            if len(parts) >= 3 and 'defective' in parts[1].lower():
                defect_folder = parts[2].lower()
                for d_name, d_idx in self.defect_map.items():
                    if d_name in defect_folder or defect_folder in d_name:
                        target = d_idx
                        break
                        
            self.samples.append((p, target))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

def get_transforms():
    train_trans = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_trans, val_trans

def build_defect_model(num_classes=len(DEFECT_CLASSES)):
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except AttributeError:
        model = models.mobilenet_v3_small(pretrained=True)
        
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def train_defect_model(epochs=5, batch_size=32, lr=1e-3):
    print("\n--- Training Stage 2: Seed Defect Type Model (MobileNetV3) ---")
    
    train_trans, val_trans = get_transforms()
    
    train_ds = SeedDefectDataset(TRAIN_DIR, transform=train_trans)
    val_ds = SeedDefectDataset(VAL_DIR, transform=val_trans)
    test_ds = SeedDefectDataset(TEST_DIR, transform=val_trans)
    
    print(f"Defect samples - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = build_defect_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data)
            total += inputs.size(0)
            
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct.double() / total if total > 0 else 0
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == targets.data)
                val_total += inputs.size(0)
                
        val_acc = val_correct.double() / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_wts)
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    test_acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    results = {
        'accuracy': float(test_acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': cm,
        'classes': DEFECT_CLASSES
    }
    
    print(f"[STAGE 2 DEFECT RESULT] Defect Test Accuracy: {test_acc*100:.2f}%, F1-score: {f1*100:.2f}%")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': DEFECT_CLASSES,
        'results': results
    }, DEFECT_MODEL_PATH)
    print(f"[SAVE] Checkpoint saved: {DEFECT_MODEL_PATH}")
    
    return results

if __name__ == '__main__':
    train_defect_model()
