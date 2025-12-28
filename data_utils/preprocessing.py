"""
Preprocessing utilities for the hybrid extractive-abstractive pipeline.

Includes:
- Sentence segmentation
- Extractive filtering using SentenceSummarizer
- SentencePiece tokenizer training and loading
- OOV handling for pointer-generator network
"""

import re
import os
import sys
from typing import List, Tuple, Optional
import sentencepiece as spm
import warnings

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.summarizer import SentenceSummarizer

# Try to import CNN-CRF model components
try:
    from src.predict import segment_text, hybrid_crf_model
    HAS_CNN_CRF = True
except Exception as e:
    HAS_CNN_CRF = False
    warnings.warn(f"CNN-CRF model not available ({e}). Using fallback regex-based segmentation.")


def segment_sentences(text: str, use_cnn_crf: bool = True) -> List[str]:
    """
    Segment text into sentences using trained CNN-CRF model.
    
    This function uses your existing trained CNN-CRF hybrid model for 
    accurate sentence boundary detection in legal documents.
    
    Args:
        text: Input text to segment
        use_cnn_crf: Whether to use CNN-CRF model (True) or fallback to regex (False)
        
    Returns:
        List of sentences
    """
    # Try to use the trained CNN-CRF model first
    if use_cnn_crf and HAS_CNN_CRF:
        try:
            # Use the hybrid CNN-CRF model for sentence segmentation
            sentences = segment_text(text, hybrid_crf_model, use_hybrid_features=True)
            
            # Clean and filter empty sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            return sentences
        except Exception as e:
            warnings.warn(f"CNN-CRF segmentation failed: {e}. Falling back to regex.")
    
    # Fallback: Basic sentence segmentation using regex
    # Split on . ! ? followed by whitespace and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def apply_extractive_filtering(
    text: str,
    top_k: int = 10,
    compression: Optional[float] = None
) -> Tuple[str, List[str], List[float]]:
    """
    Apply extractive summarization to select salient sentences.
    
    Args:
        text: Full document text
        top_k: Number of top sentences to extract
        compression: Alternative to top_k, fraction of sentences to keep
        
    Returns:
        extracted_text: Concatenated extracted sentences
        selected_sentences: List of selected sentences
        sentence_weights: Weights assigned to selected sentences
    """
    # Segment text into sentences
    sentences = segment_sentences(text)
    
    if len(sentences) == 0:
        return "", [], []
    
    # Initialize extractive summarizer
    summarizer = SentenceSummarizer(
        cnn_prob_weight=0.25,
        textrank_weight=0.35,
        tfidf_weight=0.30,
        position_weight=0.10,
        use_embeddings=False  # Disable embeddings for faster processing
    )
    
    # Apply extractive summarization
    selected_sentences, weights, _ = summarizer.summarize(
        sentences=sentences,
        original_text=text,
        top_k=top_k,
        compression=compression,
        preserve_order=True  # Keep original document order
    )
    
    # Concatenate selected sentences
    extracted_text = " ".join(selected_sentences)
    
    return extracted_text, selected_sentences, weights


def train_sentencepiece_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 50000,
    model_type: str = 'bpe',
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_piece: str = '<pad>',
    unk_piece: str = '<unk>',
    bos_piece: str = '<s>',
    eos_piece: str = '</s>'
) -> None:
    """
    Train a SentencePiece tokenizer on the given corpus.
    
    Args:
        input_file: Path to text file containing training corpus (one sentence per line)
        model_prefix: Prefix for output model files
        vocab_size: Vocabulary size
        model_type: Model type ('bpe', 'unigram', 'char', 'word')
        pad_id: ID for padding token
        unk_id: ID for unknown token
        bos_id: ID for beginning-of-sequence token
        eos_id: ID for end-of-sequence token
        pad_piece: String representation of pad token
        unk_piece: String representation of unk token
        bos_piece: String representation of bos token
        eos_piece: String representation of eos token
    """
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_piece=pad_piece,
        unk_piece=unk_piece,
        bos_piece=bos_piece,
        eos_piece=eos_piece,
        character_coverage=1.0,  # Required for languages with large character sets
        num_threads=4,
        train_extremely_large_corpus=False
    )
    
    print(f"SentencePiece model trained and saved to {model_prefix}.model")


def load_sentencepiece_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    """
    Load a trained SentencePiece tokenizer.
    
    Args:
        model_path: Path to .model file
        
    Returns:
        SentencePiece processor
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def encode_with_oov(
    text: str,
    tokenizer: spm.SentencePieceProcessor,
    vocab_size: int,
    max_len: Optional[int] = None
) -> Tuple[List[int], List[int], List[str]]:
    """
    Encode text with OOV (out-of-vocabulary) handling for pointer-generator.
    
    Creates two parallel token ID lists:
    1. Regular IDs (OOV tokens mapped to UNK)
    2. Extended IDs (OOV tokens get temporary IDs >= vocab_size)
    
    Args:
        text: Text to encode
        tokenizer: SentencePiece tokenizer
        vocab_size: Size of base vocabulary
        max_len: Maximum sequence length (if specified, will truncate)
        
    Returns:
        token_ids: Regular token IDs with OOV as UNK
        extended_ids: Extended IDs with OOV mapped to temp IDs
        oov_words: List of OOV words encountered
    """
    # Encode text to pieces
    pieces = tokenizer.encode_as_pieces(text)
    
    # Truncate if needed
    if max_len is not None:
        pieces = pieces[:max_len]
    
    # Convert pieces to IDs
    token_ids = []
    extended_ids = []
    oov_words = []
    oov_dict = {}  # Maps OOV word to temp ID
    
    for piece in pieces:
        # Get ID from vocabulary
        piece_id = tokenizer.piece_to_id(piece)
        
        # Check if it's OOV (represented as UNK)
        if piece_id == tokenizer.unk_id():
            # Check if we've seen this OOV word before
            if piece not in oov_dict:
                oov_dict[piece] = vocab_size + len(oov_words)
                oov_words.append(piece)
            
            token_ids.append(tokenizer.unk_id())
            extended_ids.append(oov_dict[piece])
        else:
            token_ids.append(piece_id)
            extended_ids.append(piece_id)
    
    return token_ids, extended_ids, oov_words


def decode_with_oov(
    token_ids: List[int],
    tokenizer: spm.SentencePieceProcessor,
    oov_words: List[str],
    vocab_size: int
) -> str:
    """
    Decode token IDs back to text, handling OOV words.
    
    Args:
        token_ids: Token IDs (may include extended IDs >= vocab_size)
        tokenizer: SentencePiece tokenizer
        oov_words: List of OOV words
        vocab_size: Size of base vocabulary
        
    Returns:
        Decoded text
    """
    pieces = []
    
    for token_id in token_ids:
        if token_id >= vocab_size:
            # This is an OOV word
            oov_idx = token_id - vocab_size
            if oov_idx < len(oov_words):
                pieces.append(oov_words[oov_idx])
        else:
            # Regular vocabulary word
            piece = tokenizer.id_to_piece(token_id)
            pieces.append(piece)
    
    # Decode pieces to text
    text = tokenizer.DecodePieces(pieces)
    return text
