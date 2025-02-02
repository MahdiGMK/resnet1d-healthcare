import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall

# ------------------- Data Loading -------------------
class ECGDataset(Dataset):
    def __init__(self, patients, results, augment=False):
        self.patients = [torch.as_tensor(p, dtype=torch.float32) for p in patients]
        self.results = [torch.as_tensor(r, dtype=torch.float32) for r in results]  # Fixed to float32
        self.augment = augment
        self.augmentor = ECGAugmentation() if augment else None

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        x = self.patients[idx].clone().detach()
        y = self.results[idx].clone().detach()

        if self.augment and self.augmentor:
            x = self.augmentor(x.unsqueeze(0)).squeeze(0)

        return x, y

class ECGAugmentation:
    def __init__(self):
        self.noise_std = 0.05
        self.scale_range = (0.9, 1.1)

    def __call__(self, x):
        # Add Gaussian noise
        x += torch.randn_like(x) * self.noise_std

        # Channel-wise scaling
        scale = torch.empty(x.size(1)).uniform_(*self.scale_range)
        x *= scale.view(1, -1, 1)

        # Random time warping
        if np.random.rand() > 0.5:
            x = self.time_warp(x)

        return x

    def time_warp(self, x, W=50):
        _, _, T = x.shape
        center = int(torch.randint(W, T-W, (1,)).item())
        warp_dist = int(torch.randint(-W, W, (1,)).item())
        warped = max(W, min(T-W, center + warp_dist))
        return torch.cat([x[:, :, :center], x[:, :, warped:]], dim=-1)

# ------------------- Enhanced Model -------------------
class AdvancedECGClassifier(nn.Module):
    def __init__(self, base_channels=64, num_layers=5, bidirectional=True):
        super().__init__()
        self.base_channels = base_channels

        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.Conv1d(4, base_channels, 15, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.ELU(),
            nn.MaxPool1d(4)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels*(2**i), base_channels*(2**(i+1)), 5)
            for i in range(num_layers)
        ])

        # Attention and LSTM
        self.attention = TemporalAttention(base_channels*(2**num_layers))
        self.lstm = nn.LSTM(
            input_size=base_channels*(2**num_layers),
            hidden_size=128,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Final classifier with depth
        lstm_out = 256 if bidirectional else 128
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 66)
        )

    def forward(self, x):
        x = self.init_conv(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.attention(x).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.elu(x + residual)

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels//4, 1)
        self.key = nn.Conv1d(channels, channels//4, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, T = x.shape
        Q = self.query(x).permute(0, 2, 1)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(torch.bmm(Q, K) / np.sqrt(C), dim=-1)
        out = torch.bmm(V, attn.permute(0, 2, 1))
        return self.gamma * out + x

# ------------------- Training Setup -------------------
def calculate_class_weights(results):
    class_counts = torch.sum(torch.stack([torch.tensor(r) for r in results]), dim=0)
    class_weights = 1.0 / (class_counts + 1e-6)
    return class_weights / class_weights.sum()

def collate_fn(batch):
    ecgs, labels = zip(*batch)
    max_length = max(ecg.shape[1] for ecg in ecgs)
    padded_ecgs = [F.pad(ecg, (0, max_length - ecg.shape[1])) for ecg in ecgs]
    return torch.stack(padded_ecgs), torch.stack(labels)

def loadhea(path):
    heafile = open(path, 'r')
    header = heafile.readline().split()
    result = {}
    result['data_name'] = header[0]
    result['lead_count'] = lead_count = int(header[1])
    result['sampling_freq'] = sampling_freq = int(header[2])
    result['total_samples'] = total_samples = int(header[3])
    result['leads'] = leads = []
    for i in range(lead_count):
        # DATA_NAME ENC ADC_RESOLUTION ADC_#DIGITS BASELINE ??? FIRST_VALUE CHECK_SUM LEAD_NAME
        lead_line = heafile.readline().split()
        [_, lead_enc, lead_adc_res, lead_adc_digs, lead_baseline, _,
            lead_first_val, lead_check_sum, lead_name] = lead_line
        leads.append({
            'enc': lead_enc,
            'adc_res': lead_adc_res,
            'adc_digs': lead_adc_digs,
            'baseline': lead_baseline,
            'first_val': lead_first_val,
            'check_sum': lead_check_sum,
            'name': lead_name
        })
    result['age'] = int(heafile.readline().split()[1])
    result['sex'] = heafile.readline().split()[1]
    result['diag'] = heafile.readline().split()[1].split(sep=',')
    return result


def pad_ecg(ecg, target_length):
    """Pad ECG to target length with zeros"""
    _, current_length = ecg.shape
    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(ecg, ((0, 0), (0, pad_width)), mode='constant')
    return ecg

def load_ecg_data(diag_mapping, data_root='../dataset/WFDBRecords', max_length=5000):
    patients = []
    results = []

    data_path = Path(data_root)
    mat_files = list(data_path.glob('**/*.mat'))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_path}")

    # Find maximum ECG length
    max_observed_length = max(
        loadmat(str(f))['val'].shape[1]
        for f in mat_files
    )
    target_length = min(max_length, max_observed_length)

    for mat_file in mat_files:
        mat_data = loadmat(str(mat_file))
        ecg_data = mat_data['val'].astype(np.float32)
        hea_file = str(mat_file)[:-3]+'hea'
        hea_data = loadhea(hea_file)
        # print(mat_file, hea_file)
        diag = (hea_data['diag'])
        # res = [diag_mapping[x] for x in diag]
        res = []
        for x in diag:
            if x in diag_mapping:
                res.append(diag_mapping[x])

        ecg_data = ecg_data[:4]


        # Pad/truncate to target length
        ecg_data = pad_ecg(ecg_data, target_length)

        # Normalize per lead
        ecg_data = (ecg_data - ecg_data.mean(axis=1, keepdims=True)) / \
                  (ecg_data.std(axis=1, keepdims=True) + 1e-8)

        patients.append(ecg_data)

        # Extract labels (modify based on your actual label extraction)
        # Example: Get labels from filename/path
        # labels = [1, 3, 5]  # Replace with actual label extraction
        results.append(res)

    # Convert labels to multi-hot encoding
    multi_hot_results = []
    for labels in results:
        encoded = np.zeros(66, dtype=np.float32)
        for label in labels:
            encoded[label-1] = 1.0  # Assuming labels are 1-6
        multi_hot_results.append(encoded)

    return patients, multi_hot_results, target_length
def train_model(diag_mapping):
    # Load data
    patients, results, _ = load_ecg_data(diag_mapping)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        patients, results, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_ds = ECGDataset(X_train, y_train, augment=True)
    val_ds = ECGDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                             collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn, num_workers=2)

    # Initialize model and metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = calculate_class_weights(results).to(device)
    model = AdvancedECGClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # Initialize metrics
    train_f1 = MultilabelF1Score(num_labels=66).to(device)
    val_f1 = MultilabelF1Score(num_labels=66).to(device)
    train_precision = MultilabelPrecision(num_labels=66).to(device)
    val_precision = MultilabelPrecision(num_labels=66).to(device)

    for epoch in range(100):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                train_f1.update(preds, targets)
                train_precision.update(preds, targets)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

                preds = torch.sigmoid(outputs)
                val_f1.update(preds, targets)
                val_precision.update(preds, targets)

        # Calculate metrics
        train_f1_score = train_f1.compute()
        val_f1_score = val_f1.compute()
        train_precision_score = train_precision.compute()
        val_precision_score = val_precision.compute()

        # Update scheduler
        scheduler.step(val_f1_score)

        # Print epoch summary
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Train F1: {train_f1_score:.4f} | Train Precision: {train_precision_score:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val F1: {val_f1_score:.4f} | Val Precision: {val_precision_score:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 50)

        # Reset metrics
        train_f1.reset()
        val_f1.reset()
        train_precision.reset()
        val_precision.reset()

if __name__ == "__main__":
    # Initialize diagnosis mapping
    diag_mapping = {}
    inv_diag_mapping = {}
    import pandas as pd
    df = pd.read_csv('../dataset/ConditionNames_SNOMED-CT.csv')
    for ind, (_, row) in enumerate(df.iterrows(), 1):
        diag_mapping[int(row['Snomed_CT'])] = ind
        diag_mapping[str(row['Snomed_CT'])] = ind
        inv_diag_mapping[ind] = row['Full Name']

    train_model(diag_mapping)
