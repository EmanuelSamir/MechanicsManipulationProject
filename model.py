import torch
import torch.nn as nn
import math

# Custom positional encoding using joint sensor measurement
class JointPositionalEncoding(nn.Module):
    def __init__(self):
        super(JointPositionalEncoding, self).__init__()

    def forward(self, torque_seq, joint_seq):
        """
        Combine torque and joint sequences with positional information.

        Args:
            torque_seq (torch.Tensor): Tensor of shape (batch_size, seq_len, feature_dim).
            joint_seq (torch.Tensor): Tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Combined tensor with positional encoding, shape (batch_size, seq_len, feature_dim).
        """
        batch_size, seq_len, feature_dim = torque_seq.size()
        
        # Generate positional indices
        positional_indices = torch.arange(0, seq_len, device=torque_seq.device).float()  # (seq_len,)
        positional_encoding = positional_indices.unsqueeze(-1).repeat(1, feature_dim)  # (seq_len, feature_dim)
        positional_encoding = positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, feature_dim)
        
        # Combine torque and joint sequences with positional encoding
        combined = torque_seq + joint_seq  # Combine torque and joint sequences
        encoded = combined * positional_encoding  # Incorporate positional information
        
        return encoded


def pad_sequences(sequences, padding_value=0.0):
    """
    Pad a list of variable-length sequences to the same length.
    
    Args:
        sequences (list of torch.Tensor): List of tensors with shapes (seq_len, feature_dim).
        padding_value (float): Value used for padding.
    
    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, max_seq_len, feature_dim).
        torch.Tensor: Mask indicating padded positions (batch_size, max_seq_len).
    """
    batch_size = len(sequences)
    max_seq_len = max(seq.size(0) for seq in sequences)
    feature_dim = sequences[0].size(1)
    
    padded_sequences = torch.full((batch_size, max_seq_len, feature_dim), padding_value)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_sequences[i, :seq_len, :] = seq
        mask[i, seq_len:] = True  # Mask out padding positions
    
    return padded_sequences, mask

# Transformer Encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, dim_model, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
    
    def forward(self, x, src_key_padding_mask=None):
        # x: (batch_size, seq_len, dim_model)
        # Transformer expects (seq_len, batch_size, dim_model)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.permute(1, 0, 2)
        return x

# Full model integrating Transformer and Feedforward
class TorquePredictionModel(nn.Module):
    def __init__(self, dim_model, num_heads, num_layers, dim_feedforward, fixed_vec_dim, output_dim, dropout=0.1):
        super(TorquePredictionModel, self).__init__()
        self.positional_encoder = JointPositionalEncoding()
        self.transformer_encoder = TransformerEncoder(dim_model, num_heads, num_layers, dim_feedforward, dropout)
        
        # Combine Transformer output with fixed-size vector
        self.fc1 = nn.Linear(dim_model + fixed_vec_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, torque_seq, fixed_vec, joint_seq, src_key_padding_mask=None):
        # Positional encoding
        torque_seq_enc = self.positional_encoder(torque_seq, joint_seq)

        # Transformer encoding
        encoded_seq = self.transformer_encoder(torque_seq_enc, src_key_padding_mask)
        
        # Aggregate encoded sequence (e.g., by averaging)
        encoded_features = encoded_seq.mean(dim=1)
        
        # Concatenate with fixed vector
        combined = torch.cat((encoded_features, fixed_vec), dim=1)
        
        # Feedforward network
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    dim_model = 6
    num_heads = 2
    num_layers = 1
    dim_feedforward = 512
    fixed_vec_dim = 6
    output_dim = 1

    batch_sequences = [
        torch.randn(5, 6),  # Sequence of length 5 with 6 features
        torch.randn(8, 6),  # Sequence of length 8 with 6 features
        torch.randn(3, 6)   # Sequence of length 3 with 6 features
    ]

    # Fixed-size input vector (batch_size, fixed_vec_dim)
    fixed_vec = torch.randn(len(batch_sequences), 6)

    # Joint measurements (same variable-length as torque_seq)
    joint_sequences = [
        torch.randn(5, 6),  # Sequence of length 5 with 6 joint features
        torch.randn(8, 6),  # Sequence of length 8 with 6 joint features
        torch.randn(3, 6)   # Sequence of length 3 with 6 joint features
    ]

    # Pad the variable-length sequences
    torque_seq, padding_mask = pad_sequences(batch_sequences)
    joint_seq, _ = pad_sequences(joint_sequences)

    model = TorquePredictionModel(dim_model, num_heads, num_layers, dim_feedforward, fixed_vec_dim, output_dim)
    # model = model.to(device)
    
    output = model.forward(torque_seq, fixed_vec, joint_seq)

    print(output)
