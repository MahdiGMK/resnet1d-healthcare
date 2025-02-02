import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder

def load_ecg_data(data_root='dataset/WFDBRecords'):
    patients = []
    results = []

    # Find all .mat files in the directory structure
    data_path = Path(data_root)
    mat_files = list(data_path.glob('**/*.mat'))  # Recursive search

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_path}")

    # Assuming directory structure encodes labels: .../class_label/patient_id/file.mat
    label_encoder = LabelEncoder()  # For converting string labels to integers

    for mat_file in mat_files:
        # Load MATLAB file
        mat_data = loadmat(str(mat_file))

        # Extract ECG data (adjust key if needed - common keys: 'val', 'ECG')
        ecg_data = mat_data['val']  # Shape should be (4, M) for 4 sensors

        # Verify data shape
        if ecg_data.shape[0] != 4:
            raise ValueError(f"Unexpected sensor count in {mat_file}: {ecg_data.shape[0]} != 4")

        # Convert to float32 and normalize
        ecg_data = ecg_data.astype(np.float32)
        ecg_data = (ecg_data - ecg_data.mean()) / (ecg_data.std() + 1e-8)

        patients.append(ecg_data)

        # Extract label from directory structure (adjust based on your actual structure)
        # Example: dataset/WFDBRecords/class_label/patient_id/file.mat
        class_label = mat_file.parent.parent.name  # Get grandparent directory name
        results.append(class_label)

    # Encode string labels to integers
    results_encoded = label_encoder.fit_transform(results)

    return patients, results_encoded, label_encoder.classes_

# Example usage:
if __name__ == "__main__":
    try:
        patients, results, class_names = load_ecg_data()
        print(f"Loaded {len(patients)} patients with {len(class_names)} classes")
        print(f"Class names: {class_names}")
        print(f"Sample ECG shape: {patients[0].shape}")
        print(f"Sample label: {results[0]}")

        # Create dataset
        dataset = ECGDataset(patients, results)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    except Exception as e:
        print(f"Error loading data: {str(e)}")

# Define the neural network architecture
class ECGClassifier(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(4, base_channels, 3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*4, 3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels*4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

# Define the Dataset class
# class ECGDataset(Dataset):
#     def __init__(self, patients, results):
#         self.patients = patients
#         self.results = results

#     def __len__(self):
#         return len(self.patients)

#     def __getitem__(self, idx):
#         # Convert to tensor and ensure correct dimensions
#         x = torch.tensor(self.patients[idx], dtype=torch.float32)
#         y = torch.tensor(self.results[idx], dtype=torch.long)
#         return x, y
class ECGDataset(Dataset):
    def __init__(self, patients, results):
        # Convert to tensors at initialization if not already tensors
        self.patients = [torch.as_tensor(p, dtype=torch.float32) for p in patients]
        self.results = [torch.as_tensor(r, dtype=torch.long) for r in results]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # Proper tensor cloning with gradient isolation
        return (
            self.patients[idx].clone().detach(),
            self.results[idx].clone().detach()
        )

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20

    # Sample data (replace with real data)
    num_patients = 1000
    time_steps = 500  # Replace with actual time series length
    patients = [torch.rand(4, time_steps) for _ in range(num_patients)]
    results = [torch.randint(0, 4, (1,)).item() for _ in range(num_patients)]

    # Create datasets and dataloaders
    dataset = ECGDataset(patients, results)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = ECGClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    print("Training complete!")
