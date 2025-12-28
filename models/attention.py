"""
Bahdanau Attention Mechanism for Pointer-Generator Network.

Implements additive attention that allows the decoder to focus on different
parts of the encoder output at each decoding step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    
    Computes attention weights and context vector based on decoder hidden state
    and encoder outputs.
    
    Args:
        encoder_hidden_dim: Dimension of encoder hidden states (including both directions)
        decoder_hidden_dim: Dimension of decoder hidden states
        attention_dim: Dimension of attention layer
    """
    
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int
    ):
        super(BahdanauAttention, self).__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim
        
        # Linear transformations for encoder outputs
        self.W_encoder = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        
        # Linear transformations for decoder hidden state
        self.W_decoder = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        
        # Linear transformation for coverage vector (if using coverage)
        self.W_coverage = nn.Linear(1, attention_dim, bias=False)
        
        # Final attention layer
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        coverage: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Current decoder hidden state [batch_size, decoder_hidden_dim]
            encoder_outputs: All encoder outputs [batch_size, seq_len, encoder_hidden_dim]
            encoder_mask: Mask for padded positions [batch_size, seq_len]
            coverage: Coverage vector from previous steps [batch_size, seq_len] or None
            
        Returns:
            context_vector: Weighted sum of encoder outputs [batch_size, encoder_hidden_dim]
            attention_weights: Attention distribution [batch_size, seq_len]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Project encoder outputs
        encoder_features = self.W_encoder(encoder_outputs)  # [batch_size, seq_len, attention_dim]
        
        # Project decoder hidden state and expand to match encoder sequence length
        decoder_features = self.W_decoder(decoder_hidden).unsqueeze(1)  # [batch_size, 1, attention_dim]
        decoder_features = decoder_features.expand(-1, seq_len, -1)  # [batch_size, seq_len, attention_dim]
        
        # Combine encoder and decoder features
        combined_features = encoder_features + decoder_features  # [batch_size, seq_len, attention_dim]
        
        # Add coverage features if provided
        if coverage is not None:
            coverage_features = self.W_coverage(coverage.unsqueeze(2))  # [batch_size, seq_len, attention_dim]
            combined_features = combined_features + coverage_features
        
        # Compute attention scores
        attention_scores = self.v(torch.tanh(combined_features)).squeeze(2)  # [batch_size, seq_len]
        
        # Mask padded positions (set to large negative value)
        attention_scores = attention_scores.masked_fill(encoder_mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Compute context vector as weighted sum of encoder outputs
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # [batch_size, encoder_hidden_dim]
        
        return context_vector, attention_weights
