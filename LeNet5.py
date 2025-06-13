import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Prepare and Process the MNIST Dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 as LeNet-5 expects
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalization for MNIST
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 2. Implement the LeNet-5 CNN
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)     # input: 1x32x32 -> output: 6x28x28
        self.pool = nn.AvgPool2d(2, 2)      # output: 6x14x14
        self.conv2 = nn.Conv2d(6, 16, 5)    # output: 16x10x10 -> 16x5x5 after pool
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))  # (64, 1, 32, 32) -> (64, 6, 28, 28) -> (64, 6, 14, 14)
        x = self.pool(torch.tanh(self.conv2(x)))  # (64, 6, 14, 14) -> (64, 16, 10, 10) -> (64, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)                # (64, 16, 5, 5) -> (64, 400)
        x = torch.tanh(self.fc1(x))               # (64, 400) -> (64, 120)
        x = torch.tanh(self.fc2(x))               # (64, 120) -> (64, 84)
        x = self.fc3(x)                           # (64, 84) -> (64, 10)
        return x
    

model = LeNet5().to(device)

# 3. Train the Neural Network
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, test_accuracies = [], []

epochs = 5
for epoch in range(epochs):
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

    train_losses.append(running_loss / len(train_loader))

    # 4. Test the Neural Network and Evaluate Performance
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

#Save model
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

