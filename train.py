import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pickle

from model import TorquePredictionModel, pad_sequences
from dataclasses import dataclass, field
import numpy as np

from matplotlib import pyplot as plt

@dataclass
class DataStream:
    sensor_stream_motion: list[tuple] = field(default_factory=list)
    joint_stream_motion: list[tuple] = field(default_factory=list)
    sensor_static: tuple = None
    distance_desired: float = None
    residual_vel: float = None

# Dummy dataset for synthetic training
class TorqueDataset(Dataset):
    def __init__(self, folder_name):
        super(TorqueDataset, self).__init__()
        torque_seqs, joint_seqs, fixed_vecs, targets = self.load_dataset(folder_name=folder_name)
        self.torque_seqs = torque_seqs
        self.joint_seqs = joint_seqs
        self.fixed_vecs = fixed_vecs
        self.targets = targets

        self.num_samples = len(self.torque_seqs)

    def load_dataset(self, folder_name):
        pkl_files = [fn for fn in os.listdir(folder_name) if fn.endswith('.pkl')]
        torque_sequences = []
        joint_sequences = []
        fixed_vecs = []
        targets = []

        for file in pkl_files:
            fp = os.path.join(folder_name, file)
            with open(fp, 'rb') as f:
                list_datastream = pickle.load(f)

            for ds in list_datastream:
                if np.isnan(ds.residual_vel):
                    continue
                torque_sequences.append(torch.tensor(ds.sensor_stream_motion)) 
                joint_sequences.append(torch.tensor(ds.joint_stream_motion))
                fixed_vecs.append(list(ds.sensor_static) + [ds.distance_desired])
                targets.append(ds.residual_vel)
        
        torque_seqs, _ = pad_sequences(torque_sequences)
        joint_seqs, _ = pad_sequences(joint_sequences)

        return torque_seqs, joint_seqs, torch.tensor(fixed_vecs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
        

    def __len__(self):
        # Returns the number of samples
        return self.num_samples

    def __getitem__(self, index):
        # Retrieves the sample corresponding to the index
        return {
            'torque_seq': self.torque_seqs[index],
            'joint_seq': self.joint_seqs[index],
            'fixed_vec': self.fixed_vecs[index],
            'target': self.targets[index]
        }
        
# Padding function
def pad_collate_fn(batch):
    torque_seqs, joint_seqs, fixed_vecs, targets = zip(*batch)
    torque_seqs_padded, padding_mask = pad_sequences(torque_seqs)
    joint_seqs_padded, _ = pad_sequences(joint_seqs)
    fixed_vecs = torch.stack(fixed_vecs)
    targets = torch.stack(targets).squeeze(-1)
    return torque_seqs_padded, joint_seqs_padded, fixed_vecs, padding_mask, targets


def plot_results(train_losses, val_losses):
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(model, batch_size=32, epochs=10, learning_rate=1e-4, device='cpu', validation_split=0.2, patience=3):
    # Initialize dataset and dataloaders
    full_dataset = TorqueDataset(folder_name='train')  # Load full dataset

    # Calculate the size of the validation set
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size

    # Split the dataset into training and validation datasets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoader for train and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("train_dataloader size:", len(train_dataloader) * batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    model.to(device)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            torque_seq = batch['torque_seq']
            joint_seq = batch['joint_seq']
            fixed_vec = batch['fixed_vec']
            targets = batch['target']

            # Move data to the device
            torque_seqs = torque_seq.to(device)
            joint_seqs = joint_seq.to(device)
            fixed_vecs = fixed_vec.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(torque_seqs, fixed_vecs, joint_seqs)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        train_losses.append(total_loss / len(train_dataloader))

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                torque_seq = batch['torque_seq']
                joint_seq = batch['joint_seq']
                fixed_vec = batch['fixed_vec']
                targets = batch['target']

                torque_seqs = torque_seq.to(device)
                joint_seqs = joint_seq.to(device)
                fixed_vecs = fixed_vec.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(torque_seqs, fixed_vecs, joint_seqs)

                # Compute validation loss
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'second_best_model.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered. Validation loss has not improved for a few epochs.")
            break

        # Switch back to training mode
        model.train()

    plot_results(train_losses, val_losses)

# Initialize model, dataset, and start training
if __name__ == '__main__':
    # Hyperparameters
    dim_model = 3
    num_heads = 1
    num_layers = 1
    dim_feedforward = 64
    fixed_vec_dim = 4
    output_dim = 1
    batch_size = 32
    epochs = 200
    learning_rate = 1e-5

    # Check device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize model and dataset
    model = TorquePredictionModel(dim_model, num_heads, num_layers, dim_feedforward, fixed_vec_dim, output_dim)

    # Train the model
    train_model(model, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
