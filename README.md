### NAME: SURYA P <br>
### REG NO: 212224230280 <br> 
### Date: 15/02/2026

## EX. No. 4 : IMPLEMENTATION OF TRANSFER LEARNING

## AIM :
To Implement Transfer Learning for classification using VGG-19 architecture.

## Problem Statement and Dataset :
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS :

### STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

### STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 3:
Train the model with training dataset.

### STEP 4:
Evaluate the model with testing dataset.

### STEP 5:
Make Predictions on New Data.

## PROGRAM :

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

```

```python
from google.colab import files
uploaded = files.upload()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

!unzip -qq ./chip_data.zip -d data
dataset_path ="./data/dataset"

train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
print("\nName: SURYA P")
print("Register Number: 212224230280")
print(f"\nTotal Training Samples: {len(train_dataset)}")
print(f"Total Testing Samples: {len(test_dataset)}")

first_image, label = train_dataset[0]
print(f"Shape of First Image: {first_image.shape}")

def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15,5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)
        image = image.numpy()
        image = (image - image.min()) / (image.max() - image.min())
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

show_sample_images(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

```

```python
model = models.vgg19(pretrained=True)
num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

for param in model.features.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0001)

```

```python
print("\nName: SURYA P")
print("Register Number: 212224230280\n")
def train_model(model, train_loader, test_loader, num_epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Validation Loss: {val_loss:.4f}")

    print("\nName: SURYA P")
    print("Register Number: 212224230280\n")
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()

train_model(model, train_loader, test_loader, num_epochs=10)

```

```python
print("\nName: SURYA P")
print("Register Number: 212224230280\n")
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes,
                cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    print("\nName: SURYA P")
    print("Register Number: 212224230280\n")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds,
                                target_names=train_dataset.classes))

test_model(model, test_loader)
```

## OUTPUT :

<img width="1586" height="450" alt="image" src="https://github.com/user-attachments/assets/78008e77-75ef-4d71-8eea-da14475c0a2e" />

### Training Loss, Validation Loss Vs Iteration Plot :

<img width="568" height="295" alt="image" src="https://github.com/user-attachments/assets/c9c27fa4-e76f-41ca-aae2-0bb98cc18e40" />

<img width="1042" height="776" alt="image" src="https://github.com/user-attachments/assets/ed831ba0-ec3d-4b5f-aea2-83ed7cdec39e" />

### Confusion Matrix :

<img width="1019" height="809" alt="image" src="https://github.com/user-attachments/assets/457b0aea-53af-4a84-bcd7-11235e4817e7" />

### Classification Report :

<img width="638" height="342" alt="image" src="https://github.com/user-attachments/assets/0a97d862-ca7b-476b-96f7-4c716f4fb112" />

### New Sample Prediction :


```python
def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    class_names = dataset.classes
    print("\nName: SURYA P")
    print("Register Number: 212224230280\n")
    image_display = image.permute(1,2,0).cpu().numpy()
    image_display = (image_display - image_display.min()) / \
                    (image_display.max() - image_display.min())

    plt.figure(figsize=(4,4))
    plt.imshow(image_display)
    plt.title(f"Actual: {class_names[label]}\n"
              f"Predicted: {class_names[predicted.item()]}")
    plt.axis("off")
    plt.show()

    print(f"Actual: {class_names[label]}")
    print(f"Predicted: {class_names[predicted.item()]}")

predict_image(model, 25, test_dataset)
predict_image(model, 55, test_dataset)

```

<img width="536" height="613" alt="image" src="https://github.com/user-attachments/assets/57c10418-aa8b-49b5-829c-f5d86716935f" />

<img width="514" height="583" alt="image" src="https://github.com/user-attachments/assets/39bc21e4-8985-44f5-a828-2460f4c8b8e9" />

## RESULT :
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
