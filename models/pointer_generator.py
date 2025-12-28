"""
Pointer-Generator Network for Abstractive Summarization.

Combines the encoder, decoder, and implements the copy mechanism that allows
the model to either generate words from the vocabulary or copy words from
the source document.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .encoder import Encoder
from .decoder import Decoder


class PointerGeneratorNetwork(nn.Module):
    """
    Complete Pointer-Generator Network with coverage mechanism.
    
    This model implements the architecture from "Get To The Point: Summarization
    with Pointer-Generator Networks" (See et al., 2017), combining:
    - BiLSTM encoder
    - LSTM decoder with Bahdanau attention
    - Copy mechanism (pointer network)
    - Coverage mechanism to reduce repetition
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        encoder_hidden_dim: Hidden dimension of encoder LSTM
        decoder_hidden_dim: Hidden dimension of decoder LSTM
        attention_dim: Dimension of attention layer
        encoder_num_layers: Number of encoder LSTM layers
        decoder_num_layers: Number of decoder LSTM layers
        dropout: Dropout probability
        pad_idx: Index of padding token
        use_coverage: Whether to use coverage mechanism
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        encoder_num_layers: int = 2,
        decoder_num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        use_coverage: bool = True
    ):
        super(PointerGeneratorNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.use_coverage = use_coverage
        self.pad_idx = pad_idx
        
        # Encoder (bidirectional, so output is 2*encoder_hidden_dim)
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=decoder_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim * 2,  # Bidirectional encoder
            attention_dim=attention_dim,
            num_layers=decoder_num_layers,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Pointer-generator mechanism
        # Computes generation probability p_gen from:
        # - context vector
        # - decoder hidden state
        # - decoder input
        self.p_gen_linear = nn.Linear(
            encoder_hidden_dim * 2 + decoder_hidden_dim + embedding_dim,
            1
        )
        
        # Share embedding weights between encoder and decoder
        self.decoder.embedding.weight = self.encoder.embedding.weight
        
    def forward(
        self,
        encoder_input: torch.Tensor,
        encoder_lengths: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_input_extended: Optional[torch.Tensor] = None,
        extra_zeros: Optional[torch.Tensor] = None,
        coverage: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            encoder_input: Source tokens [batch_size, src_len]
            encoder_lengths: Actual lengths [batch_size]
            decoder_input: Target tokens (shifted right) [batch_size, tgt_len]
            encoder_input_extended: Source tokens with OOV indices [batch_size, src_len]
            extra_zeros: Extra zeros for extended vocabulary [batch_size, max_oov]
            coverage: Initial coverage vector or None
            
        Returns:
            final_dist: Final token probability distribution [batch_size, tgt_len, extended_vocab]
            attention_weights: All attention distributions [batch_size, tgt_len, src_len]
            coverage_loss: Coverage loss for reducing repetition
        """
        batch_size = encoder_input.size(0)
        src_len = encoder_input.size(1)
        tgt_len = decoder_input.size(1)
        
        # Create encoder mask (1 for valid positions, 0 for padding)
        encoder_mask = (encoder_input != self.pad_idx).long()
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_lengths)
        # encoder_outputs: [batch_size, src_len, encoder_hidden_dim*2]
        # encoder_hidden: tuple of [num_layers, batch_size, decoder_hidden_dim]
        
        # Initialize decoder hidden state with encoder final hidden state
        decoder_hidden = encoder_hidden
        
        # Initialize coverage vector if using coverage
        if self.use_coverage and coverage is None:
            coverage = torch.zeros(batch_size, src_len).to(encoder_input.device)
        
        # Storage for outputs
        final_dists = []
        attention_dists = []
        coverage_losses = []
        
        # Decoder loop
        for t in range(tgt_len):
            # Get current decoder input token
            decoder_input_t = decoder_input[:, t:t+1]  # [batch_size, 1]
            
            # Decoder step
            vocab_logits, decoder_hidden, attention_weights, context_vector = self.decoder(
                input_token=decoder_input_t,
                hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                coverage=coverage if self.use_coverage else None
            )
            # vocab_logits: [batch_size, vocab_size]
            # attention_weights: [batch_size, src_len]
            
            # Convert logits to probabilities
            vocab_dist = F.softmax(vocab_logits, dim=1)  # [batch_size, vocab_size]
            
            # Compute generation probability p_gen
            decoder_input_emb = self.decoder.embedding(decoder_input_t).squeeze(1)  # [batch_size, embedding_dim]
            decoder_hidden_top = decoder_hidden[0][-1]  # [batch_size, decoder_hidden_dim]
            
            p_gen_input = torch.cat([
                context_vector,  # [batch_size, encoder_hidden_dim*2]
                decoder_hidden_top,  # [batch_size, decoder_hidden_dim]
                decoder_input_emb  # [batch_size, embedding_dim]
            ], dim=1)
            
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # [batch_size, 1]
            
            # Compute final distribution combining generation and copy
            # P(w) = p_gen * P_vocab(w) + (1 - p_gen) * Σ_i(a_i) for all i where x_i = w
            
            # Weight vocabulary distribution by p_gen
            weighted_vocab_dist = p_gen * vocab_dist  # [batch_size, vocab_size]
            
            # Weight attention distribution by (1 - p_gen)
            weighted_attn_dist = (1 - p_gen) * attention_weights  # [batch_size, src_len]
            
            # Add extra zeros for OOV words if needed
            if extra_zeros is not None:
                extended_vocab_dist = torch.cat([
                    weighted_vocab_dist,
                    extra_zeros
                ], dim=1)  # [batch_size, vocab_size + max_oov]
            else:
                extended_vocab_dist = weighted_vocab_dist
            
            # Scatter attention distribution to extended vocabulary
            if encoder_input_extended is not None:
                final_dist = extended_vocab_dist.scatter_add(
                    dim=1,
                    index=encoder_input_extended,
                    src=weighted_attn_dist
                )  # [batch_size, extended_vocab]
            else:
                # If no OOV handling, scatter to regular vocab
                final_dist = extended_vocab_dist.scatter_add(
                    dim=1,
                    index=encoder_input,
                    src=weighted_attn_dist
                )  # [batch_size, vocab_size]
            
            final_dists.append(final_dist)
            attention_dists.append(attention_weights)
            
            # Update coverage and compute coverage loss
            if self.use_coverage:
                # Coverage loss: sum of min(a_t, c_t) over all encoder positions
                coverage_loss = torch.sum(torch.min(attention_weights, coverage), dim=1)
                coverage_losses.append(coverage_loss)
                
                # Update coverage
                coverage = coverage + attention_weights
        
        # Stack outputs
        final_dists = torch.stack(final_dists, dim=1)  # [batch_size, tgt_len, extended_vocab]
        attention_dists = torch.stack(attention_dists, dim=1)  # [batch_size, tgt_len, src_len]
        
        # Compute total coverage loss
        if self.use_coverage and coverage_losses:
            coverage_loss = torch.stack(coverage_losses, dim=1)  # [batch_size, tgt_len]
        else:
            coverage_loss = torch.zeros(batch_size, tgt_len).to(encoder_input.device)
        
        return final_dists, attention_dists, coverage_loss
    
    def generate(
        self,
        encoder_input: torch.Tensor,
        encoder_lengths: torch.Tensor,
        encoder_input_extended: Optional[torch.Tensor] = None,
        extra_zeros: Optional[torch.Tensor] = None,
        max_len: int = 100,
        bos_idx: int = 2,
        eos_idx: int = 3
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.
        
        Args:
            encoder_input: Source tokens [batch_size, src_len]
            encoder_lengths: Actual lengths [batch_size]
            encoder_input_extended: Source with OOV indices
            extra_zeros: Extra zeros for OOV
            max_len: Maximum decode length
            bos_idx: Beginning-of-sequence token index
            eos_idx: End-of-sequence token index
            
        Returns:
            generated_ids: Generated token IDs [batch_size, max_len]
        """
        batch_size = encoder_input.size(0)
        src_len = encoder_input.size(1)
        device = encoder_input.device
        
        # Create encoder mask
        encoder_mask = (encoder_input != self.pad_idx).long()
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_lengths)
        decoder_hidden = encoder_hidden
        
        # Initialize coverage
        coverage = torch.zeros(batch_size, src_len).to(device) if self.use_coverage else None
        
        # Start with BOS token
        decoder_input = torch.full((batch_size, 1), bos_idx, dtype=torch.long).to(device)
        
        generated_ids = []
        
        for t in range(max_len):
            # Decoder step
            vocab_logits, decoder_hidden, attention_weights, context_vector = self.decoder(
                input_token=decoder_input,
                hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                coverage=coverage
            )
            
            vocab_dist = F.softmax(vocab_logits, dim=1)
            
            # Compute p_gen
            decoder_input_emb = self.decoder.embedding(decoder_input).squeeze(1)
            decoder_hidden_top = decoder_hidden[0][-1]
            
            p_gen_input = torch.cat([context_vector, decoder_hidden_top, decoder_input_emb], dim=1)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))
            
            # Compute final distribution
            weighted_vocab_dist = p_gen * vocab_dist
            weighted_attn_dist = (1 - p_gen) * attention_weights
            
            if extra_zeros is not None:
                extended_vocab_dist = torch.cat([weighted_vocab_dist, extra_zeros], dim=1)
            else:
                extended_vocab_dist = weighted_vocab_dist
            
            if encoder_input_extended is not None:
                final_dist = extended_vocab_dist.scatter_add(1, encoder_input_extended, weighted_attn_dist)
            else:
                final_dist = extended_vocab_dist.scatter_add(1, encoder_input, weighted_attn_dist)
            
            # Greedy selection
            predicted_id = torch.argmax(final_dist, dim=1, keepdim=True)  # [batch_size, 1]
            generated_ids.append(predicted_id)
            
            # Update coverage
            if self.use_coverage:
                coverage = coverage + attention_weights
            
            # Use predicted token as next input
            # If predicted ID is OOV, use UNK
            decoder_input = torch.where(
                predicted_id >= self.vocab_size,
                torch.tensor(1).to(device),  # UNK index
                predicted_id
            )
            
            # Check if all sequences have generated EOS
            if (predicted_id == eos_idx).all():
                break
        
        generated_ids = torch.cat(generated_ids, dim=1)  # [batch_size, generated_len]
        return generated_ids
