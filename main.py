import random
import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Load the pretrained AlexNet model
weights = models.AlexNet_Weights.DEFAULT
alexnet = models.alexnet(weights=weights)

# Freeze the parameters of the pretrained model
for param in alexnet.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for binary classification task
num_classes = 2  # good or bad
num_features = alexnet.classifier[6].in_features
alexnet.classifier[6] = torch.nn.Linear(num_features, num_classes)

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset using ImageFolder
data_path = 'dataset'
dataset = ImageFolder(data_path, transform=transform)

from torch.utils.data import random_split

# Split the dataset into train, validation, and test sets
train_ratio = 0.7  # 70% for training
val_ratio = 0.15  # 15% for validation
test_ratio = 0.15  # 15% for testing

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

from torch.utils.data import DataLoader

batch_size = 32

# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
alexnet = alexnet.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 100
mislabel_ratio = 0.1  # 25% mislabeled images in the training set
removal_ratio = 0.2

# Initialize lists to store loss and accuracy values
loss_values = []
accuracy_values = []
val_loss_values = []
val_accuracy_values = []

original_mr = mislabel_ratio
original_total = int(mislabel_ratio * len(train_dataset))
mislabel_indices = random.sample(range(len(train_dataset)), int(mislabel_ratio * len(train_dataset)))

for loop in range(5):
    print(f'Currently {mislabel_ratio * 100:.2f}% of images in the training set are mislabeled.')
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Training phase
        alexnet.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            if i in mislabel_indices:
                # Change the label to the opposite label
                mislabeled_labels = 1 - labels
                labels = mislabeled_labels.to(device)
            else:
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = alexnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluation phase - Training Set
        alexnet.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = alexnet(images)
                _, predicted = torch.max(outputs.data, 1)

                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        # Evaluation phase - Validation set
        alexnet.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = alexnet(images)
                _, predicted = torch.max(outputs.data, 1)

                val_loss += criterion(outputs, labels).item()
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        # Calculate the loss and accuracy for each epoch
        epoch_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        # Calculate validation loss and accuracy for the epoch
        val_epoch_loss = val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_predictions

        # Append loss and accuracy values to the lists
        loss_values.append(epoch_loss)
        accuracy_values.append(accuracy)
        val_loss_values.append(val_epoch_loss)
        val_accuracy_values.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy * 100:.2f}% - '
              f'Val Loss: {val_epoch_loss:.4f} - Val Accuracy: {val_accuracy * 100:.2f}%')

    # Generate line graphs
    epochs = range(1, num_epochs + 1)

    # Loss graph
    plt.plot(epochs, loss_values, label='Train Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.legend()
    plt.show()

    # Accuracy graph
    plt.plot(epochs, accuracy_values, label='Train Accuracy')
    plt.plot(epochs, val_accuracy_values, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Epochs vs Accuracy')
    plt.legend()
    plt.show()

    # Reset
    loss_values = []
    accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    # Remove a portion of mislabeled indices
    if mislabel_ratio > 0:
        num_indices_to_remove = int(removal_ratio * original_total)
        mislabel_indices_to_remove = random.sample(mislabel_indices, num_indices_to_remove)
        mislabel_indices = [i for i in mislabel_indices if i not in mislabel_indices_to_remove]
        mislabel_ratio -= (removal_ratio * original_mr)

        print(f'Corrected 20% of mislabeled images. Remaining mislabel ratio: {mislabel_ratio * 100:.2f}%')
