# CIFAR-10 Image Classification with CNN

A PyTorch implementation of a Convolutional Neural Network for classifying images from the CIFAR-10 dataset into 10 different categories.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Step-by-Step Implementation](#step-by-step-implementation)
  - [1. Importing Libraries](#1-importing-libraries)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Loading and Splitting Data](#3-loading-and-splitting-data)
  - [4. Building the Model](#4-building-the-model)
  - [5. Training the Model](#5-training-the-model)
  - [6. Testing the Model](#6-testing-the-model)
  - [7. Making Predictions](#7-making-predictions)
- [Results](#results)
- [Usage](#usage)
- [File Structure](#file-structure)

## 🎯 Overview
Think of this model as a **visual recognition system** that learns to identify objects in images. It takes a 32×32 color photo and outputs one of 10 categories (plane, car, bird, etc.).

This Convolutional Neural Network (CNN) can be used to classify images into 10 categories: planes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## 📊 Dataset

**CIFAR-10** - [Official Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

- **Total Images**: 60,000 color images (32×32 RGB)
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Classes**: 10 (1,000 images per class in test set)

### Classes
1. Plane
2. Car
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

--------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🏗️ Model Architecture

| Layer | Type | Input | Output | Kernel Size |
|-------|------|-------|--------|-------------|
| Conv1 | Convolutional | 3 | 12 | 5×5 |
| Pool1 | Max Pooling | - | - | 2×2 |
| Conv2 | Convolutional | 12 | 24 | 5×5 |
| Pool2 | Max Pooling | - | - | 2×2 |
| FC1 | Fully Connected | 600 | 120 | - |
| FC2 | Fully Connected | 120 | 84 | - |
| FC3 | Output | 84 | 10 | - |

**Activation Function**: ReLU

## Data Flow
<img width="2000" height="1545" alt="How-To graphic (1)" src="https://github.com/user-attachments/assets/251e012e-ac83-4f3d-834e-8cb076baeeab" />

## Core Components

### 1. Input Layer
- **What it receives**: 32×32 pixel RGB image
- **Shape**: 3 channels (Red, Green, Blue) × 32 × 32 = 3,072 values
- **Think of it as**: A grid of colored dots that the computer can read

### 2. Convolutional Layers (Feature Extractors)
**Conv Layer 1**: Detects basic patterns
- Scans image with 12 different 5×5 filters
- Each filter looks for simple features (edges, corners, colors)
- Output: 12 feature maps showing where patterns were found

**Conv Layer 2**: Detects complex patterns
- Takes the 12 feature maps and applies 24 new 5×5 filters
- Combines simple features into complex ones (eyes, wheels, wings)
- Output: 24 feature maps with higher-level patterns

### 3. Pooling Layers (Size Reducers)
- **Purpose**: Shrink the data while keeping important information
- **How**: Takes a 2×2 grid and keeps only the maximum value
- **Result**: Image size cut in half, but key features remain
- Applied after each convolutional layer

### 4. Fully Connected Layers (Decision Makers)
**FC Layer 1**: 600 → 120 neurons
- Flattens all feature maps into a single list
- Starts combining features into meaningful patterns

**FC Layer 2**: 120 → 84 neurons
- Further refines the understanding

**FC Layer 3 (Output)**: 84 → 10 neurons
- Final decision: one score for each of the 10 classes
- Highest score = predicted class

### 5. Activation Function (ReLU)
- **Purpose**: Adds non-linearity so the network can learn complex patterns
- **What it does**: Keeps positive numbers, turns negatives to zero
- Applied after every layer except the output

---------------------------------------------------------------------------------------------------------------------------------------------------------------

## 💻 Requirements

```
torch
torchvision
numpy
pillow
```

## 🔧 Installation

```bash
pip install torch torchvision numpy pillow
```


## 📝 Step-by-Step Implementation

### 1. Importing Libraries

First, import all necessary libraries for building and training the neural network:

```python
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
```

### 2. Data Preprocessing

Define transformations to normalize the images. Images are converted to tensors and normalized with mean and standard deviation of 0.5 for each RGB channel:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### 3. Loading and Splitting Data

Load the CIFAR-10 dataset and create data loaders for batch processing:

```python
# Load training and test datasets
train_data = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    transform=transform, 
    download=True
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2
)

# Define class names
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

### 4. Building the Model

Define the CNN architecture with two convolutional layers and three fully connected layers:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### 5. Training the Model

Train the model for 30 epochs using stochastic gradient descent:

```python
for epoch in range(30):
    print(f"Training epoch {epoch}...")
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Loss: {running_loss/len(train_loader):.4f}')

# Save the trained model
torch.save(net.state_dict(), 'trained_net.pth')
```

### 6. Testing the Model

Evaluate the model on the test dataset to calculate accuracy:

```python
# Load the trained model
net = NeuralNet()
net.load_state_dict(torch.load('trained_net.pth'))

correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")
```

### 7. Making Predictions

Use the trained model to predict classes for custom images:

```python
# Define transformation for custom images
new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_image(image_path):
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Predict on custom images
image_paths = ['image/cat.jpg', 'image/dog.jpg', 'image/plane.jpg']
images = [load_image(img) for img in image_paths]

net.eval()
with torch.no_grad():
    for image in images:
        output = net(image)
        _, predicted = torch.max(output, 1)
        print(f"Prediction: {class_names[predicted.item()]}")
```

## 📈 Results

The model achieves approximately **60-70% accuracy** on the CIFAR-10 test set after 30 epochs of training.

### Training Parameters
- **Epochs**: 30
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: SGD with momentum (0.9)
- **Loss Function**: Cross-Entropy Loss

## 🚀 Usage

### Training from Scratch

```bash
python main.py
```

### Using Pre-trained Model

If you have a `trained_net.pth` file:

```python
net = NeuralNet()
net.load_state_dict(torch.load('trained_net.pth'))
net.eval()
```

### Predicting Custom Images

1. Create an `image/` directory
2. Place your images in the directory
3. Update `image_paths` with your image filenames
4. Run the prediction code

## 📁 File Structure

```
.
├── main.py    # Main script
├── trained_net.pth          # Saved model weights
├── data/                    # CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-batches-py/
└── image/                   # Custom images for prediction
    ├── cat.jpg
    ├── dog.jpg
    └── plane.jpg
```
## 📄 License

MIT License

## 🙏 Acknowledgments

- CIFAR-10 dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- PyTorch framework by Meta AI
- [YouTube NeuralNine - Image Classification CNN in PyTorch]([url](https://www.youtube.com/watch?v=CtzfbUwrYGI))
---

**Note**: Make sure to create an `image/` directory with sample images before running the prediction section.
