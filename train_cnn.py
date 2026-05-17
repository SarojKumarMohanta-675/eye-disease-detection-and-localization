import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from models.cnn_model import get_model
from models.cnn_dataloader import get_dataloaders

# -----------------------
# SETTINGS
# -----------------------
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader, test_loader = get_dataloaders()
classes = train_loader.dataset.classes
num_classes = len(classes)

print("Classes:", classes)
print("Train size:", len(train_loader.dataset))
print("Val size:", len(val_loader.dataset))
print("Test size:", len(test_loader.dataset))

model = get_model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs("checkpoints", exist_ok=True)
best_val_top1 = 0.0


def topk_correct(outputs, labels, k):
    _, topk = outputs.topk(k, dim=1)
    return topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()


for epoch in range(1, EPOCHS + 1):

    # -----------------------
    # TRAIN
    # -----------------------
    model.train()
    train_total = 0
    train_top1 = 0
    train_top2 = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        train_total += bs
        train_top1 += topk_correct(outputs, labels, 1)
        train_top2 += topk_correct(outputs, labels, 2)

    train_top1_acc = 100 * train_top1 / train_total
    train_top2_acc = 100 * train_top2 / train_total

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
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

    val_top1_acc = 100 * val_top1 / val_total
    val_top2_acc = 100 * val_top2 / val_total

    scheduler.step()

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train Top-1: {train_top1_acc:.2f}% | Train Top-2: {train_top2_acc:.2f}%")
    print(f"Val   Top-1: {val_top1_acc:.2f}% | Val   Top-2: {val_top2_acc:.2f}%")

    if val_top1_acc > best_val_top1:
        best_val_top1 = val_top1_acc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("✅ Saved Best Model")


# -----------------------
# FINAL METRICS
# -----------------------

print("\n===== FINAL VALIDATION REPORT =====")
print(f"Top-1 Accuracy: {val_top1_acc:.2f}%")
print(f"Top-2 Accuracy: {val_top2_acc:.2f}%")

print("\nPrecision, Recall, F1-score:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nTraining Completed.")