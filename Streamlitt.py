import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the CIFAR-10 dataset and preprocess
transform = transforms.Compose([
        transforms.ToTensor(),  # This ensures the image is scaled to [0.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image to [-1.0, 1.0]
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model():
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0
    print('Finished Training')

# Label map for CIFAR-10
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Streamlit Interface

st.title("CIFAR-10 Image Classifier with PyTorch")
st.write("Upload an image and the model will predict the class!")

# File uploader widget for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image


    # Apply the transform to your image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((32, 32))  # Resize to match CIFAR-10 size
    image = transform(image).unsqueeze(0)  # Convert to tensor and add batch dimension
    # Convert to tensor and add batch dimension

    # Predict the class
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Show the predicted class
    st.write(f"Predicted Class: {labels[predicted_class.item()]}")

# Show a random CIFAR-10 image from the dataset for demo
if st.button('Show Random Image from CIFAR-10'):
    idx = np.random.randint(0, len(testset))
    image, label = testset[idx]
    st.image(image.permute(1, 2, 0).numpy(), caption=f"Random CIFAR-10 Image: {labels[label]}")
