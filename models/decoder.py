"""
LSTM Decoder with Attention for Pointer-Generator Network.

The decoder generates summaries one token at a time, using attention over
the encoder outputs and maintaining coverage to avoid repetition.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .attention import BahdanauAttention


class Decoder(nn.Module):
    """
    LSTM decoder with Bahdanau attention.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden states
        encoder_hidden_dim: Dimension of encoder outputs (bidirectional, so 2*encoder_hidden)
        attention_dim: Dimension of attention layer
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        pad_idx: Index of padding token
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        encoder_hidden_dim: int,
        attention_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        # Attention mechanism
        self.attention = BahdanauAttention(
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=hidden_dim,
            attention_dim=attention_dim
        )
        
        # LSTM cell - input is concatenation of embedding and context vector
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection layers
        # Combines decoder hidden state, context vector, and decoder input
        self.out = nn.Linear(
            hidden_dim + encoder_hidden_dim + embedding_dim,
            vocab_size,
            bias=True
        )
        
    def forward(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        coverage: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            input_token: Input token for this step [batch_size, 1]
            hidden: Previous hidden state tuple (h, c)
                h: [num_layers, batch_size, hidden_dim]
                c: [num_layers, batch_size, hidden_dim]
            encoder_outputs: All encoder outputs [batch_size, seq_len, encoder_hidden_dim]
            encoder_mask: Mask for encoder [batch_size, seq_len]
            coverage: Coverage vector [batch_size, seq_len] or None
            
        Returns:
            output_logits: Output logits over vocabulary [batch_size, vocab_size]
            hidden: Updated hidden state tuple
            attention_weights: Attention distribution [batch_size, seq_len]
            context_vector: Context vector from attention [batch_size, encoder_hidden_dim]
        """
        batch_size = input_token.size(0)
        
        # Embed input token
        embedded = self.embedding(input_token)  # [batch_size, 1, embedding_dim]
        embedded = self.dropout_layer(embedded)
        
        # Get decoder hidden state from LSTM hidden tuple
        # Use only the top layer for attention computation
        decoder_hidden_for_attention = hidden[0][-1]  # [batch_size, hidden_dim]
        
        # Compute attention
        context_vector, attention_weights = self.attention(
            decoder_hidden=decoder_hidden_for_attention,
            encoder_outputs=encoder_outputs,
            encoder_mask=encoder_mask,
            coverage=coverage
        )
        
        # Concatenate embedded input with context vector
        lstm_input = torch.cat([
            embedded,
            context_vector.unsqueeze(1)
        ], dim=2)  # [batch_size, 1, embedding_dim + encoder_hidden_dim]
        
        # Pass through LSTM
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        # lstm_output: [batch_size, 1, hidden_dim]
        
        # Prepare final output
        # Concatenate: LSTM output, context vector, embedded input
        output_features = torch.cat([
            lstm_output.squeeze(1),  # [batch_size, hidden_dim]
            context_vector,  # [batch_size, encoder_hidden_dim]
            embedded.squeeze(1)  # [batch_size, embedding_dim]
        ], dim=1)  # [batch_size, hidden_dim + encoder_hidden_dim + embedding_dim]
        
        # Project to vocabulary size
        output_logits = self.out(output_features)  # [batch_size, vocab_size]
        
        return output_logits, hidden, attention_weights, context_vector
