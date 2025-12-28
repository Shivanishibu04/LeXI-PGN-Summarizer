"""
FAST Training script for Pointer-Generator Network (CPU Optimized)

This script uses config_fast.py for faster training:
- 15% data sample (~1000 examples)
- Smaller model (25M vs 117M parameters)
- 10 epochs instead of 20
- Shorter sequences

Estimated time: 1-2 hours on CPU
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import pandas as pd

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config_fast as config  # Use FAST config
from models import PointerGeneratorNetwork
from data_utils import (
    SummarizationDataset,
    collate_fn,
    train_sentencepiece_tokenizer,
    load_sentencepiece_tokenizer
)
from utils import (
    LossComputer,
    MetricsTracker,
    Checkpointer,
    Timer,
    save_training_log,
    format_time
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sampled_corpus():
    """
    Prepare a SAMPLED corpus for faster tokenizer training.
    """
    print("\n" + "="*80)
    print("PREPARING SAMPLED CORPUS FOR FAST TRAINING")
    print("="*80)
    
    # Load and sample training data
    print(f"\nLoading training data from {config.TRAIN_CSV}...")
    df = pd.read_csv(config.TRAIN_CSV)
    
    # Drop missing values
    df = df.dropna(subset=['Text', 'Summary'])
    
    # Sample the data
    if config.USE_SAMPLE_DATA:
        sample_size = int(len(df) * config.SAMPLE_FRACTION)
        df = df.sample(n=sample_size, random_state=config.RANDOM_SEED)
        print(f"✓ Sampled {len(df)} examples ({config.SAMPLE_FRACTION*100:.0f}% of dataset)")
    else:
        print(f"✓ Using full dataset: {len(df)} examples")
    
    # Save sampled data for later use
    sampled_csv = config.TOKENIZER_DIR / "train_sampled.csv"
    df.to_csv(sampled_csv, index=False)
    print(f"✓ Saved sampled data to: {sampled_csv}")
    
    # Prepare corpus file (simplified - just concatenate text and summary)
    corpus_file = config.TOKENIZER_DIR / "tokenizer_corpus_fast.txt"
    print(f"\nWriting corpus to {corpus_file}...")
    
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # Write text and summary directly (skip extractive for speed)
            text = str(row['Text']).replace('\n', ' ').strip()
            summary = str(row['Summary']).replace('\n', ' ').strip()
            
            # Take first 1000 chars of text to speed up
            if len(text) > 1000:
                text = text[:1000]
            
            f.write(text + '\n')
            f.write(summary + '\n')
    
    print(f"✓ Corpus prepared: {corpus_file}")
    return str(corpus_file), str(sampled_csv)


def train_or_load_tokenizer():
    """Train or load SentencePiece tokenizer with FAST settings."""
    tokenizer_model_path = config.TOKENIZER_MODEL_FILE
    
    if os.path.exists(tokenizer_model_path):
        print(f"Loading existing tokenizer from {tokenizer_model_path}")
        tokenizer = load_sentencepiece_tokenizer(tokenizer_model_path)
    else:
        print("Tokenizer not found. Training new tokenizer (FAST mode)...")
        
        # Prepare sampled corpus
        corpus_file, _ = prepare_sampled_corpus()
        
        # Train tokenizer
        print(f"\nTraining SentencePiece tokenizer (vocab size: {config.VOCAB_SIZE})...")
        train_sentencepiece_tokenizer(
            input_file=corpus_file,
            model_prefix=config.BPE_MODEL_PREFIX,
            vocab_size=config.VOCAB_SIZE,
            model_type='bpe',
            pad_id=config.PAD_IDX,
            unk_id=config.UNK_IDX,
            bos_id=config.BOS_IDX,
            eos_id=config.EOS_IDX,
            pad_piece=config.PAD_TOKEN,
            unk_piece=config.UNK_TOKEN,
            bos_piece=config.BOS_TOKEN,
            eos_piece=config.EOS_TOKEN
        )
        
        tokenizer = load_sentencepiece_tokenizer(tokenizer_model_path)
    
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.get_piece_size()}")
    return tokenizer


def create_datasets(tokenizer):
    """Create SAMPLED datasets for faster training."""
    print("\n" + "="*80)
    print("CREATING SAMPLED DATASETS")
    print("="*80)
    
    # Check if sampled CSV exists
    sampled_csv = config.TOKENIZER_DIR / "train_sampled.csv"
    
    if sampled_csv.exists():
        csv_path = str(sampled_csv)
        print(f"Using sampled data: {csv_path}")
    else:
        # Create sample
        df = pd.read_csv(config.TRAIN_CSV)
        df = df.dropna(subset=['Text', 'Summary'])
        sample_size = int(len(df) * config.SAMPLE_FRACTION)
        df = df.sample(n=sample_size, random_state=config.RANDOM_SEED)
        sampled_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sampled_csv, index=False)
        csv_path = str(sampled_csv)
    
    # Load dataset
    full_dataset = SummarizationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        vocab_size=config.VOCAB_SIZE,
        max_encoder_len=config.MAX_ENCODER_LEN,
        max_decoder_len=config.MAX_DECODER_LEN,
        extractive_top_k=config.EXTRACTIVE_TOP_K,
        extractive_compression=config.EXTRACTIVE_COMPRESSION,
        bos_idx=config.BOS_IDX,
        eos_idx=config.EOS_IDX
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * config.VAL_SPLIT)
    train_size = dataset_size - val_size
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"\nDataset split:")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


# Import remaining functions from train.py
from train import create_model, train_epoch, validate


def main():
    """Main FAST training function."""
    print("\n" + "="*80)
    print("FAST TRAINING MODE - CPU OPTIMIZED")
    print("Hybrid Extractive-Abstractive Summarization")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Model size: ~25M parameters")
    print(f"  Dataset: {config.SAMPLE_FRACTION*100:.0f}% sample")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max encoder length: {config.MAX_ENCODER_LEN}")
    print(f"  Extractive top-k: {config.EXTRACTIVE_TOP_K}")
    print(f"  Estimated time: 1-2 hours")
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Train or load tokenizer
    tokenizer = train_or_load_tokenizer()
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=config.PAD_IDX),
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=config.PAD_IDX),
        num_workers=config.NUM_WORKERS
    )
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Create loss computer
    loss_computer = LossComputer(
        pad_idx=config.PAD_IDX,
        coverage_weight=config.COVERAGE_LOSS_WEIGHT,
        use_coverage=config.USE_COVERAGE
    )
    
    # Create checkpointer
    checkpointer = Checkpointer(config.MODEL_DIR)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_metrics, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_computer=loss_computer,
            epoch=epoch,
            global_step=global_step
        )
        
        print(f"\nTraining metrics:")
        print(f"  Total loss: {train_metrics['total_loss']:.4f}")
        print(f"  NLL loss: {train_metrics['nll_loss']:.4f}")
        print(f"  Coverage loss: {train_metrics['coverage_loss']:.4f}")
        print(f"  Time: {format_time(train_metrics['epoch_time'])}")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_computer=loss_computer
        )
        
        print(f"\nValidation metrics:")
        print(f"  Total loss: {val_metrics['total_loss']:.4f}")
        print(f"  NLL loss: {val_metrics['nll_loss']:.4f}")
        print(f"  Coverage loss: {val_metrics['coverage_loss']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        
        if is_best:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            print(f"\n✓ New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"\n  No improvement. Patience: {patience_counter}/{config.PATIENCE}")
        
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0 or is_best:
            checkpoint_path = checkpointer.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                loss=val_metrics['total_loss'],
                is_best=is_best
            )
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Log metrics
        log_entry = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss'],
            'epoch_time': train_metrics['epoch_time']
        }
        save_training_log(config.LOG_DIR / 'training_log.jsonl', log_entry)
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping after {epoch} epochs")
            break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
