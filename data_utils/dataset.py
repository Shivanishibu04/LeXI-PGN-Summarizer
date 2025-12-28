"""
PyTorch Dataset and DataLoader utilities for hybrid summarization.

Handles loading data, applying extractive filtering, tokenization,
and creating batches with proper padding and OOV handling.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
import sentencepiece as spm

from .preprocessing import apply_extractive_filtering, encode_with_oov


class SummarizationDataset(Dataset):
    """
    PyTorch Dataset for hybrid extractive-abstractive summarization.
    
    For each example:
    1. Applies extractive filtering to select salient sentences
    2. Tokenizes the extracted text (encoder input)
    3. Tokenizes the gold summary (decoder target)
    4. Handles OOV words for pointer-generator mechanism
    
    Args:
        csv_path: Path to CSV file with 'Text' and 'Summary' columns
        tokenizer: Trained SentencePiece tokenizer
        vocab_size: Size of vocabulary
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
        extractive_top_k: Number of sentences to extract
        extractive_compression: Alternative to top_k
        bos_idx: Beginning-of-sequence token index
        eos_idx: End-of-sequence token index
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer: spm.SentencePieceProcessor,
        vocab_size: int,
        max_encoder_len: int = 512,
        max_decoder_len: int = 150,
        extractive_top_k: int = 10,
        extractive_compression: Optional[float] = None,
        bos_idx: int = 2,
        eos_idx: int = 3
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.extractive_top_k = extractive_top_k
        self.extractive_compression = extractive_compression
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        
        # Load dataset
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        assert 'Text' in self.df.columns, "CSV must have 'Text' column"
        assert 'Summary' in self.df.columns, "CSV must have 'Summary' column"
        
        # Drop rows with missing values
        self.df = self.df.dropna(subset=['Text', 'Summary'])
        self.df = self.df.reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} examples")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single example.
        
        Returns a dictionary with:
        - encoder_input: Token IDs for encoder [seq_len]
        - encoder_input_extended: Extended IDs with OOV [seq_len]
        - encoder_oov_words: List of OOV words in source
        - decoder_input: Token IDs for decoder input (target shifted right) [seq_len]
        - decoder_target: Token IDs for decoder target [seq_len]
        - decoder_target_extended: Extended IDs for decoder target [seq_len]
        """
        row = self.df.iloc[idx]
        
        # Get full document and gold summary
        full_text = str(row['Text'])
        gold_summary = str(row['Summary'])
        
        # Step 1: Apply extractive filtering
        extracted_text, _, _ = apply_extractive_filtering(
            text=full_text,
            top_k=self.extractive_top_k,
            compression=self.extractive_compression
        )
        
        # Fallback to original text if extraction fails
        if not extracted_text:
            extracted_text = full_text
        
        # Step 2: Encode source (extractive output)
        encoder_ids, encoder_extended_ids, oov_words = encode_with_oov(
            text=extracted_text,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            max_len=self.max_encoder_len
        )
        
        # Step 3: Encode target (gold summary)
        # For target, we need to map any words that appear in source OOV to same extended IDs
        summary_pieces = self.tokenizer.encode_as_pieces(gold_summary)
        
        # Truncate summary if needed (leave room for BOS/EOS)
        if len(summary_pieces) > self.max_decoder_len - 2:
            summary_pieces = summary_pieces[:self.max_decoder_len - 2]
        
        # Build OOV dictionary from source
        oov_dict = {word: self.vocab_size + i for i, word in enumerate(oov_words)}
        
        # Convert summary pieces to IDs
        target_ids = []
        target_extended_ids = []
        
        for piece in summary_pieces:
            piece_id = self.tokenizer.piece_to_id(piece)
            
            # Check if piece is in source OOV
            if piece in oov_dict:
                target_ids.append(self.tokenizer.unk_id())
                target_extended_ids.append(oov_dict[piece])
            elif piece_id == self.tokenizer.unk_id():
                # OOV word not in source - just use UNK
                target_ids.append(self.tokenizer.unk_id())
                target_extended_ids.append(self.tokenizer.unk_id())
            else:
                target_ids.append(piece_id)
                target_extended_ids.append(piece_id)
        
        # Add BOS and EOS tokens
        decoder_input = [self.bos_idx] + target_ids  # BOS + target
        decoder_target = target_ids + [self.eos_idx]  # target + EOS
        decoder_target_extended = target_extended_ids + [self.eos_idx]
        
        return {
            'encoder_input': torch.tensor(encoder_ids, dtype=torch.long),
            'encoder_input_extended': torch.tensor(encoder_extended_ids, dtype=torch.long),
            'encoder_oov_words': oov_words,
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
            'decoder_target_extended': torch.tensor(decoder_target_extended, dtype=torch.long),
        }


def collate_fn(batch: List[Dict], pad_idx: int = 0) -> Dict:
    """
    Collate function for DataLoader.
    
    Pads sequences to the same length within a batch and handles
    OOV words across the batch.
    
    Args:
        batch: List of examples from SummarizationDataset
        pad_idx: Padding token index
        
    Returns:
        Dictionary with batched and padded tensors
    """
    # Extract sequences
    encoder_inputs = [item['encoder_input'] for item in batch]
    encoder_inputs_extended = [item['encoder_input_extended'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    decoder_targets = [item['decoder_target'] for item in batch]
    decoder_targets_extended = [item['decoder_target_extended'] for item in batch]
    
    # Get encoder lengths before padding
    encoder_lengths = torch.tensor([len(seq) for seq in encoder_inputs], dtype=torch.long)
    
    # Pad sequences
    encoder_input_padded = pad_sequence(
        encoder_inputs,
        batch_first=True,
        padding_value=pad_idx
    )
    
    encoder_input_extended_padded = pad_sequence(
        encoder_inputs_extended,
        batch_first=True,
        padding_value=pad_idx
    )
    
    decoder_input_padded = pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=pad_idx
    )
    
    decoder_target_padded = pad_sequence(
        decoder_targets,
        batch_first=True,
        padding_value=pad_idx
    )
    
    decoder_target_extended_padded = pad_sequence(
        decoder_targets_extended,
        batch_first=True,
        padding_value=pad_idx
    )
    
    # Find maximum OOV count in batch
    max_oov = max([len(item['encoder_oov_words']) for item in batch])
    
    # Create extra_zeros tensor for extended vocabulary
    # This will be used to extend the vocabulary distribution
    if max_oov > 0:
        batch_size = len(batch)
        extra_zeros = torch.zeros(batch_size, max_oov)
    else:
        extra_zeros = None
    
    # Collect OOV words for each example (for potential decoding later)
    oov_words_list = [item['encoder_oov_words'] for item in batch]
    
    return {
        'encoder_input': encoder_input_padded,
        'encoder_input_extended': encoder_input_extended_padded,
        'encoder_lengths': encoder_lengths,
        'decoder_input': decoder_input_padded,
        'decoder_target': decoder_target_padded,
        'decoder_target_extended': decoder_target_extended_padded,
        'extra_zeros': extra_zeros,
        'oov_words': oov_words_list
    }
