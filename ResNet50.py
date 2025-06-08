import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import pickle
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load ResNet-50
model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # freeze all layers

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 10)
model.fc.requires_grad = True
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training
epochs = 10
step_losses = []
epoch_losses = []
test_accuracies = []

print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch + 1}/{epochs}")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        step_losses.append(loss_val)

        if i % 100 == 0:
            print(f"  Step {i}/{len(train_loader)} - Loss: {loss_val:.4f}")

    avg_epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_epoch_loss)

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "resnet50_mnist.pth")
print("Model saved to resnet50_mnist.pth")

# Save metrics to .pkl
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
