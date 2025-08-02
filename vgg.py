import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import json
import os

# Device setup: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform MNIST images to fit VGG input requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),                       # Resize to 224x224 as expected by VGG
    transforms.Grayscale(num_output_channels=3),         # Convert single-channel (grayscale) to 3 channels
    transforms.ToTensor(),                               # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))                 # Normalize pixel values
])

# Load MNIST training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create data loaders for training and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load the pretrained VGG-16 model
model = vgg16(weights=VGG16_Weights.DEFAULT)

# Freeze all pretrained layers to avoid updating them
for param in model.parameters():
    param.requires_grad = False

# Replace the final classification layer with a new one for 10 MNIST classes
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
model = model.to(device)  # Move model to the selected device

# Set loss function and optimizer (only training the new classifier layer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Initialize tracking variables for loss and accuracy
train_losses = []
test_accuracies = []
step_losses = []
steps = 0
epochs = 10

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{epochs} started.")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()         # Clear gradients
        outputs = model(images)       # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()              # Backpropagation
        optimizer.step()             # Update weights (only classifier)

        running_loss += loss.item()
        step_losses.append(loss.item())
        steps += 1

        # Print loss every 100 steps
        if i % 100 == 0:
            print(f"Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Store average training loss for the epoch
    train_losses.append(running_loss / len(train_loader))

    # Evaluate model on test set
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1} complete. Avg Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save trained model weights
torch.save(model.state_dict(), "vgg16_mnist.pth")
print("Model saved to vgg16_mnist.pth")

# Save training metrics to a JSON file
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
