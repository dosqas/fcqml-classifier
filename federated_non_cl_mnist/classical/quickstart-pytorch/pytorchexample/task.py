"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, TensorDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Standard library imports
import numpy as np  # Numerical computing library for array operations

# Deep learning imports (Keras)
from keras.datasets import mnist  # MNIST handwritten digits dataset
from keras.models import Model  # Model class for creating neural networks
from keras.models import load_model
from keras.layers import Input, Dense  # Input layer and fully connected (dense) layers


class Classifier(nn.Module):
    """Enhanced classifier for 8-dimensional features"""
    def __init__(self, dropout_rate=0.2):
        super(Classifier, self).__init__()
        
        # Add batch normalization and dropout for 8D input
        self.bn_input = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.bn_input(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        
        return x


fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition of MNIST, encode with trained autoencoder, return DataLoaders."""
    # Load MNIST
    (X_train_full, y_train_full), _ = mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 784).astype("float32") / 255.0
    y_train_full = y_train_full.astype("int64")

    # Split for federated node
    num_samples = len(X_train_full)
    part_size = num_samples // num_partitions
    start = partition_id * part_size
    end = start + part_size if partition_id < num_partitions - 1 else num_samples

    # Slice partition
    X_part = X_train_full[start:end]
    y_part = y_train_full[start:end]

    # Load encoder (Keras)
    encoder = load_model(
        "/mnt/c/Users/dosqas/Documents/GitHub/fcqml-classifier/encoder/encoder_deep.keras"
    )
    encoder.trainable = False

    # ---- KERAS REQUIRES NUMPY ON CPU ----
    encoded = encoder.predict(X_part, batch_size=512)   # np array

    # ---- Convert to torch ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_part_t = torch.from_numpy(encoded).float().to(device)
    y_part_t = torch.from_numpy(y_part).long()

    # Train/Test split
    split_idx = int(0.8 * len(X_part_t))
    train_dataset = TensorDataset(X_part_t[:split_idx], y_part_t[:split_idx])
    test_dataset = TensorDataset(X_part_t[split_idx:], y_part_t[split_idx:])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load test data
    (_, _), (X_test, y_test) = mnist.load_data()

    X_test = X_test.reshape(10000, 784).astype("float32") / 255.0

    # Load encoder
    encoder = load_model(
        "/mnt/c/Users/dosqas/Documents/GitHub/fcqml-classifier/encoder/encoder_deep.keras"
    )

    # ---- Encode in NumPy ----
    encoded = encoder.predict(X_test, batch_size=512)
    encoder.trainable = False

    # ---- Convert to torch ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_test_t = torch.from_numpy(encoded).float().to(device)
    y_test_t = torch.from_numpy(y_test).long()

    dataset = TensorDataset(X_test_t, y_test_t)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    return dataloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set using Adam."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
