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

# Import Qiskit Machine Learning components
from qiskit_machine_learning.neural_networks import EstimatorQNN  # Quantum neural network wrapper

# Core Qiskit imports for building quantum circuits
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter  # Used to define trainable parameters (symbolic variables)

# Import AerSimulator for local quantum circuit simulation
from qiskit_aer import AerSimulator

# Import EstimatorV2 (modern replacement for Estimator primitive) for circuit evaluation
from qiskit_ibm_runtime import EstimatorV2

# Connector to bridge Qiskit QNNs with PyTorch modules
from qiskit_machine_learning.connectors import TorchConnector

# SparsePauliOp defines quantum observables (like measuring Pauli Z)
from qiskit.quantum_info import SparsePauliOp

# Gradient computation method using parameter-shift rule
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient


class HybridQuantumClassifier(nn.Module):
    """Minimal 8-qubit quantum classifier for speed"""
    def __init__(self, dropout_rate=0.2):
        super(HybridQuantumClassifier, self).__init__()
        
        # Minimal quantum circuit (8 qubits)
        self.quantum_layer = self._build_fast_circuit()
        
        # Minimal classical post-processing
        self.fc1 = nn.Linear(8, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(32, 10)
        
        # Initialize weights
        self._initialize_weights()

    def _build_fast_circuit(self):
        """Simplest possible 8-qubit circuit"""
        n_qubits = 8
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        
        # Input parameters (8 features)
        input_params = [Parameter(f'x{i}') for i in range(n_qubits)]
        
        # Minimal trainable parameters (8 only)
        theta_params = [Parameter(f't{i}') for i in range(n_qubits)]
        
        # SIMPLEST encoding: just RY rotations
        for i in range(n_qubits):
            qc.ry(input_params[i], i)
        
        # Single trainable rotation per qubit
        for i in range(n_qubits):
            qc.rz(theta_params[i], i)
        
        # NO entanglement for maximum speed
        # If you want entanglement, uncomment ONE of these lines:
        # qc.cx(0, 1)  # Just one CNOT
        # OR
        # for i in range(0, n_qubits-1, 2):  # Pairwise
        #     qc.cx(i, i+1)
        
        # Create observables (measure Z on each qubit)
        observables = []
        for i in range(n_qubits):
            # Create string like "IIIIZIII" where Z is at position i
            obs_str = ['I'] * n_qubits
            obs_str[i] = 'Z'
            observables.append(SparsePauliOp(''.join(obs_str)))
        
        # Use statevector simulator (fastest for small circuits)
        backend = AerSimulator(method='statevector')
        
        # Import EstimatorV2 correctly
        from qiskit_ibm_runtime import EstimatorV2
        estimator = EstimatorV2(mode=backend)
        
        # Create QNN
        qnn = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            input_params=input_params,
            weight_params=theta_params,
            observables=observables,
            # Disable gradient for speed if needed
            # gradient=None
        )
        
        return TorchConnector(qnn)

    def _initialize_weights(self):
        """Initialize classical layers"""
        for m in [self.fc1, self.fc2]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Quantum processing (8D -> 8D)
        x = self.quantum_layer(x)
        
        # Classical processing
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Output
        x = self.fc2(x)
        
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
