import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load dataset
train_data = datasets.ImageFolder('chest_xray_data/train', transform=transform)
test_data = datasets.ImageFolder('chest_xray_data/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# model setup
model = models.resnet18(pretrained=True)

# modify final layer for 2 classes (Normal vs Pneumonia)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# loss + optimizer
class_weights = torch.tensor([1.94, 0.67]).to(device) # dataset is imbalanced, 3x more pneumonia than normal
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training loop
num_epochs = 10
best_accuracy = 0
best_epoch = 0
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # training
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    running_loss /= len(train_loader)

    # evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Loss: {running_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print("\n")

    train_losses.append(running_loss)
    test_accuracies.append(accuracy)

    # save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model.pth")

print(f"Best model saved with accuracy {best_accuracy:.4f} at epoch {best_epoch}")

# plot training loss and test accuracy curves
plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.show()

plt.figure()
plt.plot(test_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Curve")
plt.show()


#--------------------------------
# Results:

# Epoch 1/10
# Loss: 0.0954
# Accuracy: 0.8862, Precision: 0.8521, Recall: 0.9897

# Epoch 2/10
# Loss: 0.0306
# Accuracy: 0.8269, Precision: 0.7843, Recall: 0.9974

# Epoch 3/10
# Loss: 0.0143
# Accuracy: 0.8205, Precision: 0.7769, Recall: 1.0000

# Epoch 4/10
# Loss: 0.0129
# Accuracy: 0.7708, Precision: 0.7317, Recall: 1.0000

# Epoch 5/10
# Loss: 0.0111
# Accuracy: 0.8526, Precision: 0.8143, Recall: 0.9897

# Epoch 6/10
# Loss: 0.0078
# Accuracy: 0.8189, Precision: 0.7764, Recall: 0.9974

# Epoch 7/10
# Loss: 0.0065
# Accuracy: 0.8686, Precision: 0.8277, Recall: 0.9974

# Epoch 8/10
# Loss: 0.0062
# Accuracy: 0.8510, Precision: 0.8087, Recall: 0.9974

# Epoch 9/10
# Loss: 0.0072
# Accuracy: 0.8141, Precision: 0.7718, Recall: 0.9974

# Epoch 10/10
# Loss: 0.0019
# Accuracy: 0.8269, Precision: 0.7831, Recall: 1.0000

# Best model saved with accuracy 0.8862 at epoch 1