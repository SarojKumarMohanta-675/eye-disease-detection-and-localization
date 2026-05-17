import os
import tempfile
import cv2
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from ultralytics import YOLO

from models.cnn_model import get_model

# Paths
CNN_MODEL_PATH = "checkpoints/best_model.pth"
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"
TRAIN_FOLDER = "dataset/split/train"
IMG_SIZE = 224

st.title("👁 Eye Disease Detection System")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # Save temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    image.save(temp_path)

    # ---------------- CNN ----------------
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

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        top2 = torch.topk(probs, 2)

    top1_idx = top2.indices[0].item()
    top2_idx = top2.indices[1].item()

    st.subheader("🧠 CNN Classification Result")
    st.write("Top-1:", classes[top1_idx])
    st.write("Top-2:", classes[top2_idx])

    # ---------------- YOLO ----------------
    st.subheader("📦 YOLO Detection Result")

    yolo = YOLO(YOLO_MODEL_PATH)
    results = yolo.predict(source=temp_path, imgsz=320, conf=0.10)

    # Load original image (no color change)
    img = cv2.imread(temp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{yolo.names[cls]} {conf:.2f}"

            # 🔴 Draw thick bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

            # 🔴 Bigger label with background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)

            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 30

            # Background rectangle
            cv2.rectangle(
                img,
                (x1, text_y - text_h - 10),
                (x1 + text_w, text_y),
                (255, 0, 0),
                -1
            )

            # White text
            cv2.putText(
                img,
                label,
                (x1, text_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3
            )

        st.image(img, caption="Detection Output", width="stretch")

    else:
        st.image(img, caption="Detection Output (No detection)", width="stretch")
        st.warning("No bounding box detected.")