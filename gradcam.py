#gradcam implementation file

# imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2 # to use for heatmap overlay

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # convert grayscale to 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# for visualization
# NO normalization so images look normal
display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# load test dataset
test_data = datasets.ImageFolder('chest_xray_data/test', transform=transform)
display_data = datasets.ImageFolder('chest_xray_data/test', transform=display_transform)

test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

class_names = test_data.classes

# load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False) # have to match training
model.fc = nn.Linear(model.fc.in_features, 2)

# load saved best model
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# gradcam class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # forward hook to store feature maps
        self.target_layer.register_forward_hook(self.save_activations)

        # backward hook to store gradients
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)

        # select class score
        score = output[:, class_idx]
        score.backward()

        # get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]

        # global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))

        # weighted sum of feature maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # apply ReLU
        # only positive influence
        cam = torch.relu(cam)

        # normalize between 0 and 1
        cam -= cam.min()
        cam /= cam.max()

        return cam.cpu().detach().numpy()

# visualization
def show_gradcam(idx):
    """
    shows original image, heatmap, and overlay
    """

    # get image
    input_img, label = test_data[idx]
    display_img, _ = display_data[idx]

    input_tensor = input_img.unsqueeze(0).to(device)

    # prediction
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

    confidence = torch.softmax(output, dim=1)[0][pred].item()

    print(f"True: {class_names[label]}, Pred: {class_names[pred]}, Confidence: {confidence:.3f}")

    # generate gradcam
    cam = gradcam.generate_cam(input_tensor, pred)

    img = display_img.permute(1, 2, 0).numpy()

    cam = cv2.resize(cam, (224, 224))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = heatmap * 0.4 + np.uint8(img * 255)

    correct = "Correct" if pred == label else "Incorrect"

    plt.figure(figsize=(12, 4))

    # original
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original\nTrue: {class_names[label]}")
    plt.axis("off")

    # heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    # overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay.astype(np.uint8))
    plt.title(f"Pred: {class_names[pred]} ({correct})\nConf: {confidence:.2f}")
    plt.axis("off")

    plt.tight_layout()

    # save images
    plt.savefig(f"gradcam_{idx}_{correct}.png")
    plt.close()

# find examples by class
def find_class_examples(target_class, num_samples=5):
    indices = []

    class_idx = class_names.index(target_class)

    for i in range(len(test_data)):
        _, label = test_data[i]

        if label == class_idx:
            indices.append(i)

        if len(indices) >= num_samples:
            break

    return indices

# initialize gradcam
target_layer = model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)

# run analysis
normal_idxs = find_class_examples("NORMAL", 5)
pneumonia_idxs = find_class_examples("PNEUMONIA", 5)

print("\nNORMAL CASES:")
for idx in normal_idxs:
    show_gradcam(idx)

print("\nPNEUMONIA CASES:")
for idx in pneumonia_idxs:
    show_gradcam(idx)
