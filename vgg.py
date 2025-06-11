import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import json
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform MNIST to match VGG input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG expects 224x224
    transforms.Grayscale(num_output_channels=3),  # Make it 3-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load pre-trained VGG-16
model = vgg16(weights=VGG16_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # Freeze backbone

# Replace classifier
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # 10 MNIST classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Tracking
train_losses = []
test_accuracies = []
step_losses = []
steps = 0
epochs = 10

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{epochs} started.")
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step_losses.append(loss.item())
        steps += 1

        if i % 100 == 0:
            print(f"Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_losses.append(running_loss / len(train_loader))

    # Evaluation
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
    print(f"Epoch {epoch+1} complete. Avg Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "vgg16_mnist.pth")
print("Model saved to vgg16_mnist.pth")

# Save metrics
with open("vgg16_training_stats.json", "w") as f:
    json.dump({
        "train_losses": train_losses,
        "step_losses": step_losses,
        "test_accuracies": test_accuracies
    }, f)
print("Training stats saved to vgg16_training_stats.json")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(step_losses, label="Step Loss", alpha=0.5)
plt.plot(
    [i * len(train_loader) for i in range(len(train_losses))],
    train_losses, label="Epoch Loss", linewidth=2)
plt.title("Loss Over Time")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label="Test Accuracy", color='orange')
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.savefig("vgg16_training_plot.png")
plt.show()
