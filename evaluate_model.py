import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from models.cnn_model import get_model
from models.cnn_dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader, test_loader = get_dataloaders()
classes = train_loader.dataset.classes
num_classes = len(classes)

# Load model
model = get_model(num_classes)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

def topk_correct(outputs, labels, k):
    _, topk = outputs.topk(k, dim=1)
    return topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()

val_total = 0
val_top1 = 0
val_top2 = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        bs = labels.size(0)
        val_total += bs
        val_top1 += topk_correct(outputs, labels, 1)
        val_top2 += topk_correct(outputs, labels, 2)

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

top1 = 100 * val_top1 / val_total
top2 = 100 * val_top2 / val_total

print("\n===== FINAL VALIDATION REPORT =====")
print(f"Top-1 Accuracy: {top1:.2f}%")
print(f"Top-2 Accuracy: {top2:.2f}%")

print("\nPrecision, Recall, F1-score:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))