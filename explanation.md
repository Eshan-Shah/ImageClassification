Model Training Workflow (LeNet5, VGG16, ResNet50)
This document explains the core logic behind the three image classification models you implemented — LeNet5.py, vgg.py, and ResNet50.py — using a 7-stage deep learning training cycle. Each stage is described with how it appears in code.

1. Device Setup
Selects GPU if available, otherwise uses CPU.

python
Copy
Edit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
2. Data Preparation & Transformation
All models train on the MNIST dataset but need different transformations to match their expected input formats.

LeNet5
python
Copy
Edit
transforms.Resize((32, 32)),
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
VGG16
python
Copy
Edit
transforms.Resize((224, 224)),
transforms.Grayscale(num_output_channels=3),  # Converts 1-channel to 3-channel
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
ResNet50
python
Copy
Edit
transforms.Resize((64, 64)),
transforms.Grayscale(num_output_channels=3),
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
3. Model Configuration
Each model is either custom (LeNet5) or pretrained (VGG, ResNet50), with the final classification layer replaced.

LeNet5
python
Copy
Edit
class LeNet5(nn.Module):
    ...
model = LeNet5().to(device)
VGG16
python
Copy
Edit
model = vgg16(weights=VGG16_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # Freeze pretrained layers
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
model = model.to(device)
ResNet50
python
Copy
Edit
model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
4. Loss Function and Optimizer
Cross-entropy loss is used for multi-class classification. Only the trainable parameters (final layers) are passed to the optimizer in pretrained models.

python
Copy
Edit
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.<final_layer>.parameters(), lr=0.001)
Example for VGG:

python
Copy
Edit
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
5. Training Loop
Handles gradient computation, backpropagation, and weight updates.

python
Copy
Edit
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
Some scripts also track per-step losses:

python
Copy
Edit
step_losses.append(loss.item())
6. Evaluation Loop
Switches model to evaluation mode and computes accuracy.

python
Copy
Edit
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
7. Saving Outputs
Each model is saved to disk, and training metrics are optionally logged in .json or .pkl.

LeNet5
python
Copy
Edit
torch.save(model.state_dict(), "lenet5_mnist.pth")
VGG16
python
Copy
Edit
torch.save(model.state_dict(), "vgg16_mnist.pth")
with open("vgg16_training_stats.json", "w") as f:
    json.dump({...}, f)
ResNet50
python
Copy
Edit
torch.save(model.state_dict(), "resnet50_mnist.pth")
with open("metrics.pkl", "wb") as f:
    pickle.dump({...}, f)
All three also generate loss/accuracy plots with matplotlib and save them as .png files.

Summary Table
Component	LeNet5	VGG16	ResNet50
Input Size	32×32 (1ch)	224×224 (3ch)	64×64 (3ch)
Model Type	Custom CNN	Pretrained (modified FC)	Pretrained (modified FC)
Frozen Layers	N/A	All except final layer	All except final layer
Outputs	.pth, .png	.pth, .json, .png	.pth, .pkl, .png

