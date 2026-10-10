import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from backend.config import DL_MODEL_PATH, PROCESSED_DATA_DIR, CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE, MODELS_DIR

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

def build_mobilenet_v3(num_classes=len(CLASSES)):
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(pretrained=True)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

class DLEngine:
    def __init__(self, model_path=DL_MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device(DEVICE)
        self.model = None

    def train_and_evaluate(self, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
        print("--- Starting MobileNetV3 Transfer Learning Pipeline ---")
        train_transform, val_test_transform = get_transforms()

        train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
        val_dir = os.path.join(PROCESSED_DATA_DIR, 'val')
        test_dir = os.path.join(PROCESSED_DATA_DIR, 'test')

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = build_mobilenet_v3(num_classes=len(CLASSES)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss, running_corrects, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)

            epoch_loss = running_loss / max(1, total)
            epoch_acc = running_corrects / max(1, total)

            # Validation
            model.eval()
            val_loss, val_corrects, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data).item()
                    val_total += inputs.size(0)

            val_acc = val_corrects / max(1, val_total)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss/max(1,val_total):.4f} Val Acc: {val_acc:.4f}")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.model_path)
                print(f"--> Saved best model checkpoint to {self.model_path}")

        # Load best model for Test Evaluation
        print("--- Evaluating MobileNetV3 on TEST Dataset ---")
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)

        results = {
            'MobileNetV3': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'confusion_matrix': cm,
                'classification_report': report
            }
        }

        metrics_file = os.path.join(MODELS_DIR, 'dl_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"MobileNetV3 Test Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        self.model = model
        return results

    def predict_image(self, image_input):
        """
        Inference function for a single seed crop (PIL Image or BGR numpy array).
        Returns predicted class and confidence.
        """
        if self.model is None:
            self.model = build_mobilenet_v3(num_classes=len(CLASSES)).to(self.device)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()

        if isinstance(image_input, np.ndarray):
            # Convert BGR to RGB PIL Image
            rgb = torch.tensor(image_input[:, :, ::-1].copy())
            pil_img = Image.fromarray(image_input[:, :, ::-1])
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            pil_img = Image.open(image_input).convert('RGB')

        _, transform = get_transforms()
        tensor_img = transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probs, dim=0)

        return {
            'class': CLASSES[pred_idx.item()],
            'confidence': float(confidence.item()),
            'model_used': 'MobileNetV3'
        }
