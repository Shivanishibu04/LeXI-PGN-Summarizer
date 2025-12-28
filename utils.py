"""
Utility functions for training and evaluation.

Includes:
- Loss computation with coverage
- Training metrics tracking
- Model checkpointing
- Validation routines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import time
import json
from pathlib import Path


class LossComputer:
    """
    Computes the combined loss for pointer-generator network.
    
    Combines:
    - Negative log-likelihood loss (NLL) for token prediction
    - Coverage loss to reduce repetition
    """
    
    def __init__(
        self,
        pad_idx: int = 0,
        coverage_weight: float = 1.0,
        use_coverage: bool = True
    ):
        """
        Args:
            pad_idx: Padding token index (to ignore in loss)
            coverage_weight: Weight for coverage loss term
            use_coverage: Whether to use coverage mechanism
        """
        self.pad_idx = pad_idx
        self.coverage_weight = coverage_weight
        self.use_coverage = use_coverage
        
    def compute_loss(
        self,
        final_dist: torch.Tensor,
        target: torch.Tensor,
        coverage_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            final_dist: Final token probability distribution [batch_size, seq_len, vocab_size]
            target: Target token IDs [batch_size, seq_len]
            coverage_loss: Coverage loss per position [batch_size, seq_len] or None
            
        Returns:
            total_loss: Combined loss (scalar)
            loss_dict: Dictionary with individual loss components
        """
        batch_size, seq_len, vocab_size = final_dist.size()
        
        # Reshape for loss computation
        final_dist = final_dist.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        target = target.view(-1)  # [batch_size * seq_len]
        
        # Create mask for non-padding positions
        mask = (target != self.pad_idx).float()
        
        # Compute negative log-likelihood loss
        # Gather the probabilities for the target tokens
        target_probs = torch.gather(final_dist, dim=1, index=target.unsqueeze(1)).squeeze(1)
        
        # Avoid log(0) by adding small epsilon
        target_probs = target_probs + 1e-10
        
        # Compute NLL
        nll_loss = -torch.log(target_probs)
        
        # Apply mask and compute mean
        nll_loss = (nll_loss * mask).sum() / mask.sum()
        
        # Initialize total loss with NLL
        total_loss = nll_loss
        
        # Add coverage loss if available
        avg_coverage_loss = torch.tensor(0.0).to(nll_loss.device)
        if self.use_coverage and coverage_loss is not None:
            # Reshape coverage loss
            coverage_loss = coverage_loss.view(-1)  # [batch_size * seq_len]
            
            # Apply mask and compute mean
            avg_coverage_loss = (coverage_loss * mask).sum() / mask.sum()
            
            # Add to total loss
            total_loss = total_loss + self.coverage_weight * avg_coverage_loss
        
        # Prepare loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'nll_loss': nll_loss.item(),
            'coverage_loss': avg_coverage_loss.item() if isinstance(avg_coverage_loss, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict


class MetricsTracker:
    """
    Tracks training and validation metrics.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.nll_loss = 0.0
        self.coverage_loss = 0.0
        self.num_batches = 0
        
    def update(self, loss_dict: Dict[str, float]):
        """Update metrics with new batch."""
        self.total_loss += loss_dict['total_loss']
        self.nll_loss += loss_dict['nll_loss']
        self.coverage_loss += loss_dict['coverage_loss']
        self.num_batches += 1
        
    def get_average(self) -> Dict[str, float]:
        """Get average metrics."""
        if self.num_batches == 0:
            return {
                'total_loss': 0.0,
                'nll_loss': 0.0,
                'coverage_loss': 0.0
            }
        
        return {
            'total_loss': self.total_loss / self.num_batches,
            'nll_loss': self.nll_loss / self.num_batches,
            'coverage_loss': self.coverage_loss / self.num_batches
        }


class Checkpointer:
    """
    Handles model checkpointing.
    """
    
    def __init__(self, checkpoint_dir: Path, keep_best_n: int = 3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.best_checkpoints = []  # List of (loss, path) tuples
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        is_best: bool = False
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            loss: Current loss
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # If this is a best checkpoint, add to list
        if is_best:
            self.best_checkpoints.append((loss, checkpoint_path))
            self.best_checkpoints.sort(key=lambda x: x[0])  # Sort by loss
            
            # Keep only best N checkpoints
            if len(self.best_checkpoints) > self.keep_best_n:
                _, old_checkpoint_path = self.best_checkpoints.pop()
                if old_checkpoint_path.exists():
                    old_checkpoint_path.unlink()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


class Timer:
    """Simple timer for tracking training time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0.0
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time
            self.start_time = None
        return self.elapsed_time
    
    def get_elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed_time


def save_training_log(log_file: Path, log_entry: Dict):
    """
    Append training log entry to file.
    
    Args:
        log_file: Path to log file
        log_entry: Dictionary with log data
    """
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
