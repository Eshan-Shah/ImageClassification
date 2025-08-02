import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import pickle
import os

# Device setup: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms for MNIST (ResNet expects 3-channel, 64x64 images)
transform = transforms.Compose([
    transforms.Resize((64, 64)),                         # Resize from 28x28 to 64x64
    transforms.Grayscale(num_output_channels=3),         # Convert grayscale to 3-channel image
    transforms.ToTensor(),                               # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))                 # Normalize pixel values
])

# Load MNIST training and testing datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create data loaders to load data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load pretrained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze all layers so their weights won't be updated during training
for param in model.parameters():
    param.requires_grad = False

# Replace the layer to match 10 MNIST classes
model.fc = nn.Linear(model.fc.in_features, 10)
model.fc.requires_grad = True  # Ensure the new layer is trainable
model = model.to(device)       # Move model to selected device

# Define loss function and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop setup
epochs = 10
step_losses = []        # Stores loss after every training step
epoch_losses = []       # Stores average loss per epoch
test_accuracies = []    # Stores test accuracy after each epoch

print("Starting training...")
for epoch in range(epochs):
    model.train()       # Set model to training mode
    running_loss = 0.0
    print(f"\nEpoch {epoch + 1}/{epochs}")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Clear gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                # Backward pass
        optimizer.step()               # Update only the fc layer

        loss_val = loss.item()
        running_loss += loss_val
        step_losses.append(loss_val)

        # Optional: print progress every 100 steps
        if i % 100 == 0:
            print(f"  Step {i}/{len(train_loader)} - Loss: {loss_val:.4f}")

    # Compute average loss for the entire epoch
    avg_epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_epoch_loss)

    # Evaluation on the test set
    model.eval()   # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient tracking for inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# Save trained model weights to file
torch.save(model.state_dict(), "resnet50_mnist.pth")
print("Model saved to resnet50_mnist.pth")

# Save training metrics for later analysis
with open("metrics.pkl", "wb") as f:
    pickle.dump({
        "step_losses": step_losses,
        "epoch_losses": epoch_losses,
        "test_accuracies": test_accuracies
    }, f)
print("Metrics saved to metrics.pkl")

# Plot: Loss vs Step
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(step_losses, label="Loss per step", color="blue")
plt.title("Loss vs Step")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()

# Plot: Loss vs Epoch
plt.subplot(1, 2, 2)
plt.plot(epoch_losses, label="Avg Loss per epoch", marker='o', color="green")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("loss_plots.png")
plt.show()
print("Loss plots saved to loss_plots.png")
