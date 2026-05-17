import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.cnn_model import get_model

# ---------------- SETTINGS ----------------
IMG_PATH = r"dataset/split/train/Diabetic Retinopathy/DR1.jpg"
MODEL_PATH = r"checkpoints/best_model.pth"
IMG_SIZE = 224
THRESH = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load classes
dataset = ImageFolder("dataset/split/train")
classes = dataset.classes

# Load model
model = get_model(len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Hooks
activations = None
gradients = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Load image
img = Image.open(IMG_PATH).convert("RGB")
img_np = np.array(img)
h, w, _ = img_np.shape

input_tensor = transform(img).unsqueeze(0).to(device)

# Forward
output = model(input_tensor)
pred_class = output.argmax().item()
print("Predicted:", classes[pred_class])

# Backward
model.zero_grad()
output[0, pred_class].backward()

# Grad-CAM
act = activations.detach()
grad = gradients.detach()

weights = grad.mean(dim=(2, 3), keepdim=True)
cam = (weights * act).sum(dim=1).squeeze()
cam = torch.relu(cam).cpu().numpy()

cam = (cam - cam.min()) / (cam.max() + 1e-8)

# Resize to original size
cam = cv2.resize(cam, (w, h))
cam_uint8 = (cam * 255).astype(np.uint8)

# Threshold to create mask
_, mask = cv2.threshold(cam_uint8, int(THRESH * 255), 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Bounding box
if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
else:
    # fallback
    x, y, bw, bh = int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6)

# Draw bounding box
img_box = img_np.copy()
cv2.rectangle(img_box, (x, y), (x + bw, y + bh), (0, 255, 0), 3)

# Save output
cv2.imwrite("gradcam_bbox.jpg", cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR))

print("Saved: gradcam_bbox.jpg")