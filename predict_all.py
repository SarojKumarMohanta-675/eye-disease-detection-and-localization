import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from ultralytics import YOLO

from models.cnn_model import get_model

# ----------------------------
# PATHS (DO NOT CHANGE)
# ----------------------------
CNN_MODEL_PATH = "checkpoints/best_model.pth"
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"
TRAIN_FOLDER = "dataset/split/train"
IMG_SIZE = 224


def cnn_predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolder(TRAIN_FOLDER)
    classes = dataset.classes
    num_classes = len(classes)

    model = get_model(num_classes)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        top2 = torch.topk(probs, 2)

    top1_idx = top2.indices[0].item()
    top2_idx = top2.indices[1].item()

    print("\n🔎 CNN Classification Result")
    print("Top-1:", classes[top1_idx])
    print("Top-2:", classes[top2_idx])


def yolo_detect(image_path):
    print("\n📦 YOLO Detection Running...")
    model = YOLO(YOLO_MODEL_PATH)
    model.predict(source=image_path, imgsz=320, conf=0.25, save=True)
    print("Detection image saved in runs/detect/predict/")


if __name__ == "__main__":
    image_path = input("Enter image path (example: demo_images/test.jpg): ").strip()

    if not os.path.exists(image_path):
        print("❌ Image not found.")
    else:
        cnn_predict(image_path)
        yolo_detect(image_path)