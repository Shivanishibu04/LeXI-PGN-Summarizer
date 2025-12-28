"""
Configuration file for the Hybrid Extractive-Abstractive Summarization Pipeline.

This module contains all hyperparameters and settings for the Pointer-Generator Network
and the training pipeline.
"""

import torch
from pathlib import Path

# ========================
# Path Configuration
# ========================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "summariser_dataset"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "pgn_output"
MODEL_DIR = OUTPUT_DIR / "models"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, TOKENIZER_DIR, LOG_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================
# Extractive Summarization Config
# ========================
EXTRACTIVE_TOP_K = 10  # Number of sentences to extract per document
EXTRACTIVE_COMPRESSION = None  # Alternative to top_k (e.g., 0.3 for 30% compression)

# ========================
# Tokenization Config
# ========================
VOCAB_SIZE = 50000  # SentencePiece vocabulary size
BPE_MODEL_PREFIX = str(TOKENIZER_DIR / "legal_bpe")
TOKENIZER_MODEL_FILE = f"{BPE_MODEL_PREFIX}.model"
TOKENIZER_VOCAB_FILE = f"{BPE_MODEL_PREFIX}.vocab"

# Special tokens
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

# ========================
# Model Architecture Config
# ========================
# Encoder
EMBEDDING_DIM = 256
ENCODER_HIDDEN_DIM = 512
ENCODER_NUM_LAYERS = 2
ENCODER_DROPOUT = 0.3
ENCODER_BIDIRECTIONAL = True

# Decoder
DECODER_HIDDEN_DIM = 512
DECODER_NUM_LAYERS = 1
DECODER_DROPOUT = 0.3

# Attention
ATTENTION_DIM = 512

# Sequence limits
MAX_ENCODER_LEN = 512  # Maximum input sequence length
MAX_DECODER_LEN = 150  # Maximum output sequence length

# ========================
# Training Config
# ========================
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
GRAD_CLIP = 5.0  # Gradient clipping threshold

# Loss weights
COVERAGE_LOSS_WEIGHT = 1.0  # Weight for coverage loss component
USE_COVERAGE = True  # Whether to use coverage mechanism

# Validation
VAL_SPLIT = 0.1  # Fraction of training data to use for validation
EVAL_EVERY_N_STEPS = 500  # Evaluate on validation set every N steps
SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs

# Early stopping
PATIENCE = 3  # Number of epochs without improvement before stopping

# ========================
# Device Config
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # DataLoader workers (set to 0 for Windows compatibility)

# ========================
# Logging Config
# ========================
LOG_INTERVAL = 100  # Log training metrics every N batches
VERBOSE = True

# ========================
# Inference Config
# ========================
BEAM_SIZE = 4  # Beam size for beam search during inference
MIN_DECODE_LEN = 35  # Minimum summary length during inference
MAX_DECODE_LEN = 120  # Maximum summary length during inference

# ========================
# Reproducibility
# ========================
RANDOM_SEED = 42

print(f"Configuration loaded. Device: {DEVICE}")
print(f"Training data: {TRAIN_CSV}")
print(f"Test data: {TEST_CSV}")
print(f"Output directory: {OUTPUT_DIR}")
