"""
Quick validation script to verify all components are working correctly.

This script performs sanity checks on:
1. Dataset loading
2. Tokenizer training
3. Model initialization
4. Data pipeline
5. Forward pass
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from models import PointerGeneratorNetwork
from data_utils import (
    train_sentencepiece_tokenizer,
    load_sentencepiece_tokenizer,
    SummarizationDataset,
    collate_fn
)
from torch.utils.data import DataLoader


def check_dataset():
    """Verify dataset files exist and are valid."""
    print("\n" + "="*80)
    print("CHECKING DATASET")
    print("="*80)
    
    train_csv = config.TRAIN_CSV
    test_csv = config.TEST_CSV
    
    # Check files exist
    assert train_csv.exists(), f"Training data not found: {train_csv}"
    assert test_csv.exists(), f"Test data not found: {test_csv}"
    print(f"✓ Training data found: {train_csv}")
    print(f"✓ Test data found: {test_csv}")
    
    # Load and check structure
    train_df = pd.read_csv(train_csv, nrows=5)
    assert 'Text' in train_df.columns, "Missing 'Text' column"
    assert 'Summary' in train_df.columns, "Missing 'Summary' column"
    print(f"✓ Dataset columns: {train_df.columns.tolist()}")
    print(f"✓ Sample loaded successfully")
    
    # Check sizes
    train_df_full = pd.read_csv(train_csv)
    test_df_full = pd.read_csv(test_csv)
    print(f"✓ Training examples: {len(train_df_full)}")
    print(f"✓ Test examples: {len(test_df_full)}")
    
    return True


def check_tokenizer():
    """Verify tokenizer can be trained/loaded."""
    print("\n" + "="*80)
    print("CHECKING TOKENIZER")
    print("="*80)
    
    # Create a small corpus for testing
    test_corpus_file = config.TOKENIZER_DIR / "test_corpus.txt"
    
    with open(test_corpus_file, 'w', encoding='utf-8') as f:
        f.write("This is a test sentence for tokenizer training.\n")
        f.write("Legal documents require special tokenization.\n")
        f.write("The Pointer-Generator Network uses BPE tokenization.\n")
    
    print(f"✓ Created test corpus: {test_corpus_file}")
    
    # Train a small tokenizer
    test_prefix = str(config.TOKENIZER_DIR / "test_bpe")
    print(f"Training test tokenizer...")
    
    train_sentencepiece_tokenizer(
        input_file=str(test_corpus_file),
        model_prefix=test_prefix,
        vocab_size=1000,  # Small vocab for testing
        model_type='bpe'
    )
    
    print(f"✓ Tokenizer trained successfully")
    
    # Load and test
    tokenizer = load_sentencepiece_tokenizer(f"{test_prefix}.model")
    
    test_text = "This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"✓ Tokenizer loaded successfully")
    print(f"  Original: {test_text}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    
    # Clean up test files
    os.remove(test_corpus_file)
    os.remove(f"{test_prefix}.model")
    os.remove(f"{test_prefix}.vocab")
    
    return tokenizer


def check_model():
    """Verify model can be initialized."""
    print("\n" + "="*80)
    print("CHECKING MODEL")
    print("="*80)
    
    model = PointerGeneratorNetwork(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        encoder_hidden_dim=config.ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=config.DECODER_HIDDEN_DIM,
        attention_dim=config.ATTENTION_DIM,
        encoder_num_layers=config.ENCODER_NUM_LAYERS,
        decoder_num_layers=config.DECODER_NUM_LAYERS,
        dropout=config.ENCODER_DROPOUT,
        pad_idx=config.PAD_IDX,
        use_coverage=config.USE_COVERAGE
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {config.DEVICE}")
    
    return model


def check_forward_pass(model):
    """Verify forward pass works."""
    print("\n" + "="*80)
    print("CHECKING FORWARD PASS")
    print("="*80)
    
    batch_size = 2
    src_len = 20
    tgt_len = 10
    max_oov = 5
    
    # Create dummy inputs
    encoder_input = torch.randint(0, config.VOCAB_SIZE, (batch_size, src_len))
    encoder_input_extended = encoder_input.clone()
    encoder_lengths = torch.tensor([src_len, src_len - 5])
    decoder_input = torch.randint(0, config.VOCAB_SIZE, (batch_size, tgt_len))
    extra_zeros = torch.zeros(batch_size, max_oov)
    
    print(f"✓ Created dummy batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Source length: {src_len}")
    print(f"  Target length: {tgt_len}")
    
    # Move to device
    model = model.to(config.DEVICE)
    encoder_input = encoder_input.to(config.DEVICE)
    encoder_input_extended = encoder_input_extended.to(config.DEVICE)
    encoder_lengths = encoder_lengths.to(config.DEVICE)
    decoder_input = decoder_input.to(config.DEVICE)
    extra_zeros = extra_zeros.to(config.DEVICE)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        final_dist, attention_weights, coverage_loss = model(
            encoder_input=encoder_input,
            encoder_lengths=encoder_lengths,
            decoder_input=decoder_input,
            encoder_input_extended=encoder_input_extended,
            extra_zeros=extra_zeros
        )
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {final_dist.shape}")
    print(f"  Attention shape: {attention_weights.shape}")
    print(f"  Coverage loss shape: {coverage_loss.shape}")
    
    # Check shapes
    expected_vocab = config.VOCAB_SIZE + max_oov
    assert final_dist.shape == (batch_size, tgt_len, expected_vocab), "Unexpected output shape"
    assert attention_weights.shape == (batch_size, tgt_len, src_len), "Unexpected attention shape"
    print(f"✓ Output shapes correct")
    
    return True


def main():
    """Run all validation checks."""
    print("\n" + "="*80)
    print("VALIDATION SCRIPT FOR POINTER-GENERATOR NETWORK")
    print("="*80)
    
    try:
        # Check 1: Dataset
        check_dataset()
        
        # Check 2: Tokenizer
        check_tokenizer()
        
        # Check 3: Model
        model = check_model()
        
        # Check 4: Forward pass
        check_forward_pass(model)
        
        # All checks passed
        print("\n" + "="*80)
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("="*80)
        print("\nYour setup is ready for training!")
        print("\nNext steps:")
        print("1. Run 'python example.py' to see the extractive component in action")
        print("2. Run 'python train.py' to train the full model")
        print("3. Run 'python inference.py' to generate summaries")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ VALIDATION FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
