import torch
from torch import nn
import torch.nn.functional as F
import math


class HyenaFilter(nn.Module):
    """
    Hyena Filter: Implements the implicit long convolution via FFT.
    This is a key component that makes Hyena efficient for long sequences.
    """
    def __init__(self, d_model, seq_len, order=2, filter_order=64):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.order = order
        
        # Learnable filter parameters
        self.filter_fn = nn.Sequential(
            nn.Linear(filter_order, d_model * order),
            nn.ReLU(),
            nn.Linear(d_model * order, d_model * order)
        )
        
        # Position encodings for filter
        self.register_buffer('positions', torch.linspace(0, 1, filter_order))
        
    def forward(self, x):
        """
        Args:
            x: (batch, d_model, seq_len)
        Returns:
            filtered: (batch, d_model, seq_len)
        """
        batch_size = x.shape[0]
        
        # Generate filters
        filters = self.filter_fn(self.positions.unsqueeze(0))  # (1, d_model * order)
        filters = filters.view(1, self.d_model, self.order)  # (1, d_model, order)
        
        # FFT-based convolution
        x_fft = torch.fft.rfft(x, n=self.seq_len, dim=-1)
        
        # Apply filters in frequency domain
        output = torch.zeros_like(x)
        for i in range(self.order):
            filter_weight = filters[:, :, i].unsqueeze(-1)
            filtered_fft = x_fft * filter_weight
            output += torch.fft.irfft(filtered_fft, n=self.seq_len, dim=-1)
            
        return output


class HyenaOperator(nn.Module):
    """
    Hyena Operator: Subquadratic drop-in replacement for attention.
    Uses data-controlled gating and implicit long convolutions.
    """
    def __init__(self, d_model, seq_len, order=2, filter_order=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.order = order
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * (order + 1))
        
        # Hyena filters
        self.filters = nn.ModuleList([
            HyenaFilter(d_model, seq_len, order, filter_order)
            for _ in range(order)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, d_model, seq_len)
        Returns:
            output: (batch, d_model, seq_len)
        """
        batch_size, d_model, seq_len = x.shape
        
        # Transpose for linear projection: (batch, seq_len, d_model)
        x_proj = x.transpose(1, 2)
        
        # Project input to multiple branches
        projections = self.in_proj(x_proj)  # (batch, seq_len, d_model * (order + 1))
        projections = projections.view(batch_size, seq_len, self.order + 1, d_model)
        
        # Split into v and gates
        v = projections[:, :, 0, :].transpose(1, 2)  # (batch, d_model, seq_len)
        gates = [projections[:, :, i+1, :].transpose(1, 2) for i in range(self.order)]
        
        # Apply Hyena recurrence
        output = v
        for i, (gate, filter_layer) in enumerate(zip(gates, self.filters)):
            # Element-wise gating
            output = output * gate
            # Long convolution via filter
            output = filter_layer(output)
        
        # Transpose back and project output
        output = output.transpose(1, 2)  # (batch, seq_len, d_model)
        output = self.out_proj(output)
        output = self.dropout(output)
        output = output.transpose(1, 2)  # (batch, d_model, seq_len)
        
        return output


class HyenaBlock(nn.Module):
    """
    Hyena Block: Combines Hyena operator with feed-forward network.
    Similar to Transformer block but uses Hyena instead of attention.
    """
    def __init__(self, d_model, seq_len, order=2, filter_order=64, 
                 ff_mult=4, dropout=0.1):
        super().__init__()
        
        self.hyena = HyenaOperator(d_model, seq_len, order, filter_order, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_model * ff_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * ff_mult, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, d_model, seq_len)
        Returns:
            output: (batch, d_model, seq_len)
        """
        # Hyena with residual
        x = x + self.hyena(self.norm1(x))
        
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class scEpiLock_Hyena(nn.Module):
    """
    scEpiLock with Hyena operators for efficient long-range modeling.
    Uses Hyena's subquadratic long convolutions instead of standard convolutions.
    """
    def __init__(self, n_class, seq_len=1000, d_model=320, n_layers=4, 
                 hyena_order=2, dropout=0.2):
        super(scEpiLock_Hyena, self).__init__()
        
        # Initial embedding convolution
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=d_model, kernel_size=8),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Calculate sequence length after embedding
        embed_seq_len = seq_len - 7  # After conv with kernel_size=8
        
        # Stack of Hyena blocks
        self.hyena_blocks = nn.ModuleList([
            HyenaBlock(
                d_model=d_model,
                seq_len=embed_seq_len,
                order=hyena_order,
                filter_order=64,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Pooling and classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 925),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(925, n_class)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, 4, seq_len) - one-hot encoded DNA sequences
        Returns:
            output: (batch, n_class) - class predictions
        """
        # Embed sequences
        x = self.embedding(x)  # (batch, d_model, seq_len-7)
        
        # Apply Hyena blocks
        for block in self.hyena_blocks:
            x = block(x)
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        x = self.dropout(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


class scEpiLock_HyenaDeep(nn.Module):
    """
    Deep scEpiLock with Hyena operators and hierarchical processing.
    Combines convolutions for local patterns with Hyena for long-range dependencies.
    """
    def __init__(self, n_class, seq_len=1000):
        super(scEpiLock_HyenaDeep, self).__init__()
        
        # Local feature extraction with convolutions
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8),
            nn.BatchNorm1d(320),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2)
        )
        
        # Calculate sequence length after initial convs and pooling
        seq_after_conv = ((seq_len - 7) - 7) // 4  # ~246 for seq_len=1000
        
        # Hyena blocks for long-range interactions
        self.hyena1 = HyenaBlock(d_model=480, seq_len=seq_after_conv, order=2, dropout=0.2)
        self.hyena2 = HyenaBlock(d_model=480, seq_len=seq_after_conv, order=2, dropout=0.2)
        
        # Further processing
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=480, out_channels=960, kernel_size=4),
            nn.BatchNorm1d(960),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.5)
        )
        
        # Calculate final sequence length
        seq_final = (seq_after_conv - 3) // 4  # ~60
        
        # Final Hyena block
        self.hyena3 = HyenaBlock(d_model=960, seq_len=seq_final, order=2, dropout=0.5)
        
        # Global pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(960, 925),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(925, n_class)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, 4, seq_len) - one-hot encoded DNA sequences
        Returns:
            output: (batch, n_class) - class predictions
        """
        # Local feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Long-range modeling with Hyena
        x = self.hyena1(x)
        x = self.hyena2(x)
        
        # Further processing
        x = self.conv3(x)
        x = self.hyena3(x)
        
        # Global pooling and classification
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x
