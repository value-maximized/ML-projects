# %%
# Use Case: to classify images into several classes (planes, cars, birds, et.)
# Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
# Description: 
# - 60K color images (RGB) with 32x32 across 10 classes (planes, cars, birds, et.)
# - Dataset split into 50K training images (of 5 test batches of 10K images - 1K images per class ) and 10K test images  
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# %%
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=32,shuffle=True,num_workers=2)

# %%
image, label = train_data[0]
image.size()
# 3 channels for RGB, 32px x 32px

# %%
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# %%
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,12,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(12,24,5)
        self.fc1 = nn.Linear(24*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

# %%
for epoch in range(30):
    print(f"Training epoch {epoch}...")
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Loss: {running_loss/len(train_loader):.4f}')

# %%
torch.save(net.state_dict(),'trained_net.pth')
net = NeuralNet()
net.load_state_dict(torch.load('trained_net.pth'))

# %%
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

# %%

new_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def load_image(image_path):
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0)
    return image
image_paths = ['image/cat.jpg', 'image/dog.jpg','image/plane.jpg']
images = [load_image(img) for img in image_paths]
net.eval()
with torch.no_grad():
    for image in images:
        output = net(image)
        _, predicted = torch.max(output,1)
        print(f"Prediction: {class_names[predicted.item()]}")


