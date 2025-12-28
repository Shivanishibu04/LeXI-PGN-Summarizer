"""
Training script for Pointer-Generator Network.

This script implements the complete training pipeline:
1. Loads or trains SentencePiece tokenizer
2. Creates training and validation datasets with extractive filtering
3. Initializes Pointer-Generator Network
4. Trains the model with coverage loss
5. Validates and saves checkpoints
6. Logs training metrics
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
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


def train_or_load_tokenizer():
    """
    Train SentencePiece tokenizer or load existing one.
    
    Returns:
        SentencePiece processor
    """
    tokenizer_model_path = config.TOKENIZER_MODEL_FILE
    
    if os.path.exists(tokenizer_model_path):
        print(f"Loading existing tokenizer from {tokenizer_model_path}")
        tokenizer = load_sentencepiece_tokenizer(tokenizer_model_path)
    else:
        print("Tokenizer not found. Training new tokenizer...")
        
        # First, prepare the corpus
        from prepare_data import prepare_tokenizer_corpus
        corpus_file = prepare_tokenizer_corpus()
        
        # Train tokenizer
        print(f"\nTraining SentencePiece tokenizer...")
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
    """
    Create training and validation datasets.
    
    Args:
        tokenizer: SentencePiece tokenizer
        
    Returns:
        train_dataset, val_dataset
    """
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)
    
    # Load full training dataset
    full_dataset = SummarizationDataset(
        csv_path=str(config.TRAIN_CSV),
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
    
    # Create random split
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


def create_model():
    """
    Initialize Pointer-Generator Network.
    
    Returns:
        Model instance
    """
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
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
    
    # Move to device
    model = model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {config.DEVICE}")
    
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_computer: LossComputer,
    epoch: int,
    global_step: int
) -> tuple:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_computer: Loss computation utility
        epoch: Current epoch number
        global_step: Global training step
        
    Returns:
        avg_metrics, global_step
    """
    model.train()
    metrics_tracker = MetricsTracker()
    timer = Timer()
    timer.start()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        encoder_input = batch['encoder_input'].to(config.DEVICE)
        encoder_input_extended = batch['encoder_input_extended'].to(config.DEVICE)
        encoder_lengths = batch['encoder_lengths'].to(config.DEVICE)
        decoder_input = batch['decoder_input'].to(config.DEVICE)
        decoder_target_extended = batch['decoder_target_extended'].to(config.DEVICE)
        
        extra_zeros = batch['extra_zeros']
        if extra_zeros is not None:
            extra_zeros = extra_zeros.to(config.DEVICE)
        
        # Forward pass
        final_dist, attention_weights, coverage_loss = model(
            encoder_input=encoder_input,
            encoder_lengths=encoder_lengths,
            decoder_input=decoder_input,
            encoder_input_extended=encoder_input_extended,
            extra_zeros=extra_zeros,
            coverage=None
        )
        
        # Compute loss
        total_loss, loss_dict = loss_computer.compute_loss(
            final_dist=final_dist,
            target=decoder_target_extended,
            coverage_loss=coverage_loss
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        optimizer.step()
        
        # Update metrics
        metrics_tracker.update(loss_dict)
        global_step += 1
        
        # Update progress bar
        if batch_idx % config.LOG_INTERVAL == 0:
            avg_metrics = metrics_tracker.get_average()
            progress_bar.set_postfix({
                'loss': f"{avg_metrics['total_loss']:.4f}",
                'nll': f"{avg_metrics['nll_loss']:.4f}",
                'cov': f"{avg_metrics['coverage_loss']:.4f}"
            })
    
    elapsed_time = timer.stop()
    avg_metrics = metrics_tracker.get_average()
    avg_metrics['epoch_time'] = elapsed_time
    
    return avg_metrics, global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_computer: LossComputer
) -> dict:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_computer: Loss computation utility
        
    Returns:
        Average validation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Move batch to device
            encoder_input = batch['encoder_input'].to(config.DEVICE)
            encoder_input_extended = batch['encoder_input_extended'].to(config.DEVICE)
            encoder_lengths = batch['encoder_lengths'].to(config.DEVICE)
            decoder_input = batch['decoder_input'].to(config.DEVICE)
            decoder_target_extended = batch['decoder_target_extended'].to(config.DEVICE)
            
            extra_zeros = batch['extra_zeros']
            if extra_zeros is not None:
                extra_zeros = extra_zeros.to(config.DEVICE)
            
            # Forward pass
            final_dist, attention_weights, coverage_loss = model(
                encoder_input=encoder_input,
                encoder_lengths=encoder_lengths,
                decoder_input=decoder_input,
                encoder_input_extended=encoder_input_extended,
                extra_zeros=extra_zeros,
                coverage=None
            )
            
            # Compute loss
            total_loss, loss_dict = loss_computer.compute_loss(
                final_dist=final_dist,
                target=decoder_target_extended,
                coverage_loss=coverage_loss
            )
            
            # Update metrics
            metrics_tracker.update(loss_dict)
    
    return metrics_tracker.get_average()


def main():
    """Main training function."""
    print("\n" + "="*80)
    print("HYBRID EXTRACTIVE-ABSTRACTIVE SUMMARIZATION TRAINING")
    print("Pointer-Generator Network with Coverage Mechanism")
    print("="*80)
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    print(f"\nRandom seed: {config.RANDOM_SEED}")
    
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
            'train_nll': train_metrics['nll_loss'],
            'train_coverage': train_metrics['coverage_loss'],
            'val_loss': val_metrics['total_loss'],
            'val_nll': val_metrics['nll_loss'],
            'val_coverage': val_metrics['coverage_loss'],
            'epoch_time': train_metrics['epoch_time']
        }
        save_training_log(config.LOG_DIR / 'training_log.jsonl', log_entry)
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*80}")
            break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model checkpoints saved to: {config.MODEL_DIR}")
    print(f"Training logs saved to: {config.LOG_DIR}")


if __name__ == "__main__":
    main()
