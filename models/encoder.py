"""
BiLSTM Encoder for Pointer-Generator Network.

The encoder processes the extractively-filtered input text and produces
contextualized representations for each token.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder that processes input sequences.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden states
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        pad_idx: Index of padding token for embedding layer
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer with padding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Linear layers to reduce bidirectional hidden states to unidirectional
        # This allows compatibility with the unidirectional decoder
        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            input_lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            encoder_outputs: Contextualized token representations [batch_size, seq_len, hidden_dim*2]
            hidden: Tuple of (h_n, c_n) reduced to [num_layers, batch_size, hidden_dim]
        """
        batch_size = input_ids.size(0)
        
        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout_layer(embedded)
        
        # Pack padded sequences for efficient LSTM processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )  # [batch_size, seq_len, hidden_dim*2]
        
        # Reduce bidirectional hidden states to unidirectional
        # hidden: [num_layers*2, batch_size, hidden_dim] -> [num_layers, batch_size, hidden_dim]
        # cell:   [num_layers*2, batch_size, hidden_dim] -> [num_layers, batch_size, hidden_dim]
        
        # Concatenate forward and backward hidden states for each layer
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        
        # Concatenate forward and backward, then reduce
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # [num_layers, batch_size, hidden_dim*2]
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)  # [num_layers, batch_size, hidden_dim*2]
        
        # Apply reduction
        hidden = self.reduce_h(hidden)  # [num_layers, batch_size, hidden_dim]
        cell = self.reduce_c(cell)  # [num_layers, batch_size, hidden_dim]
        
        return encoder_outputs, (hidden, cell)
