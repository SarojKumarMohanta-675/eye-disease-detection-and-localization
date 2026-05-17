import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths
RAW_DIR = "dataset/original_raw"
SPLIT_DIR = "dataset/split"

TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

# Create split folders if not exist
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)

# Split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Loop through each class
for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    # First split: train and temp
    train_images, temp_images = train_test_split(
        images, test_size=(1 - train_ratio), random_state=42
    )

    # Second split: val and test
    val_images, test_images = train_test_split(
        temp_images, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
    )

    # Create class folders inside train/val/test
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(VAL_DIR, class_name, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(TEST_DIR, class_name, img)
        )

    print(f"{class_name} → Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

print("Dataset split completed successfully!")