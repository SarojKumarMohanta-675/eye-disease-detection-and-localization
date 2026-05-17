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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load class names
dataset = ImageFolder("dataset/split/train")
classes = dataset.classes

# Load trained model
model = get_model(len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Variables for hooks
activations = None
gradients = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Hook last convolution block of EfficientNet
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Load image
img = Image.open(IMG_PATH).convert("RGB")
img_np = np.array(img)

# Prepare input
input_tensor = transform(img).unsqueeze(0).to(device)

# Forward pass
output = model(input_tensor)
pred_class = output.argmax().item()
print("Predicted class:", classes[pred_class])

# Backward pass
model.zero_grad()
output[0, pred_class].backward()

# Grad-CAM calculation
act = activations.detach()
grad = gradients.detach()

weights = grad.mean(dim=(2, 3), keepdim=True)
cam = (weights * act).sum(dim=1).squeeze()
cam = torch.relu(cam).cpu().numpy()

# Normalize heatmap
cam = (cam - cam.min()) / (cam.max() + 1e-8)

# Resize heatmap to original image size
cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
heatmap = np.uint8(255 * cam)

# Apply color map
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Convert original image to BGR for OpenCV blending
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# Overlay heatmap on original image
overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

# Save output
cv2.imwrite("gradcam_output.jpg", overlay)

print("Saved: gradcam_output.jpg")