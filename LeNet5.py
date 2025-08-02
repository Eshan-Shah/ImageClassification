import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration: Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Prepare and Process the MNIST Dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 as required by LeNet-5
    transforms.ToTensor(),        # Convert PIL images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST dataset mean and std
])

# Download and load the training and testing datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create data loaders to feed data in batches
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 2. Define the LeNet-5 CNN architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)     # First convolutional layer
        self.pool = nn.AvgPool2d(2, 2)      # Average pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)    # Second convolutional layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # First fully connected layer
        self.fc2 = nn.Linear(120, 84)          # Second fully connected layer
        self.fc3 = nn.Linear(84, 10)           # Output layer for 10 classes

    def forward(self, x):
        # Apply conv -> tanh -> pool operations
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)            # Flatten the feature map
        x = torch.tanh(self.fc1(x))          # Fully connected layer with tanh activation
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)                      # Final output layer (raw scores for each class)
        return x

# Initialize the model and move it to the selected device
model = LeNet5().to(device)

# 3. Set up loss function and optimizer for training
criterion = nn.CrossEntropyLoss()                   # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

train_losses, test_accuracies = [], []

# Train the model for a number of epochs
epochs = 5
for epoch in range(epochs):
    model.train()                    # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()       # Clear previous gradients
        outputs = model(images)     # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()             # Backpropagation
        optimizer.step()            # Update weights

        running_loss += loss.item() # Accumulate loss

    train_losses.append(running_loss / len(train_loader))  # Average loss for the epoch

    # 4. Evaluate model on test data
    model.eval()                    # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():          # Disable gradient calculation for testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save the trained model to a file
torch.save(model.state_dict(), "lenet5_mnist.pth")
print("Model saved as lenet5_mnist.pth")


# 5. Visualize Results
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

