import sys
import os

# Go one folder up so Python can find the "models" folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.cnn_model import get_model

# -------------------------
# SETTINGS
# -------------------------
SPLIT_DIR = os.path.join(PROJECT_ROOT, "dataset", "split")
OUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "yolo_det")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

IMAGE_SIZE = 224
THRESH = 0.35

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Load class names from train split
train_folder = os.path.join(SPLIT_DIR, "train")
dataset_train = ImageFolder(train_folder)
classes = dataset_train.classes
num_classes = len(classes)
print("Classes:", classes)

# Load trained CNN model
model = get_model(num_classes)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.to(device)
model.eval()

activations = None
gradients = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Hook last feature block
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def yolo_xywh_from_box(x, y, bw, bh, w, h):
    xc = (x + bw / 2) / w
    yc = (y + bh / 2) / h
    bw_n = bw / w
    bh_n = bh / h

    xc = float(max(0.0, min(1.0, xc)))
    yc = float(max(0.0, min(1.0, yc)))
    bw_n = float(max(0.0, min(1.0, bw_n)))
    bh_n = float(max(0.0, min(1.0, bh_n)))

    return xc, yc, bw_n, bh_n

def process_split(split_name: str):
    src_dir = os.path.join(SPLIT_DIR, split_name)
    out_img_dir = os.path.join(OUT_DIR, "images", split_name)
    out_lbl_dir = os.path.join(OUT_DIR, "labels", split_name)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    dataset = ImageFolder(src_dir)
    print(f"\nProcessing {split_name} images: {len(dataset)}")

    for img_path, label in dataset.samples:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        input_tensor = transform(img).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)

        outputs = model(input_tensor)
        score = outputs[0, label]
        score.backward()

        act = activations.detach()
        grad = gradients.detach()

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        cam_uint8 = (cam_resized * 255).astype(np.uint8)

        # Threshold CAM
        _, mask = cv2.threshold(cam_uint8, int(THRESH * 255), 255, cv2.THRESH_BINARY)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)

            # Fallback if box too small
            if bw < 5 or bh < 5:
                x, y, bw, bh = int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)
        else:
            # Fallback if no contour found
            x, y, bw, bh = int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)

        xc, yc, bw_n, bh_n = yolo_xywh_from_box(x, y, bw, bh, w, h)

        base = os.path.splitext(os.path.basename(img_path))[0]
        img_out_path = os.path.join(out_img_dir, base + ".jpg")
        lbl_out_path = os.path.join(out_lbl_dir, base + ".txt")

        img.save(img_out_path, quality=95)

        with open(lbl_out_path, "w", encoding="utf-8") as f:
            f.write(f"{label} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")

    print(f"{split_name} done.")

if __name__ == "__main__":
    process_split("train")
    process_split("val")
    process_split("test")
    print("\n✅ Bounding boxes generated successfully into:", OUT_DIR)