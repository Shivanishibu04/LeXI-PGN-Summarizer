"""
FAST Configuration for CPU Training - Optimized for Speed

This configuration reduces training time by:
1. Using smaller dataset (10% sample)
2. Smaller model architecture
3. Fewer epochs
4. Smaller batch size
5. Shorter sequences

Estimated time: 1-2 hours total on CPU
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
OUTPUT_DIR = PROJECT_ROOT / "pgn_output_fast"
MODEL_DIR = OUTPUT_DIR / "models"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, TOKENIZER_DIR, LOG_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================
# DATA SAMPLING (Reduce dataset size)
# ========================
USE_SAMPLE_DATA = True  # Use only a subset of data
SAMPLE_FRACTION = 0.15  # Use 15% of data (~1000 examples instead of 7000)

# ========================
# Extractive Summarization Config
# ========================
EXTRACTIVE_TOP_K = 5  # REDUCED from 10 (shorter sequences)
EXTRACTIVE_COMPRESSION = None

# ========================
# Tokenization Config
# ========================
VOCAB_SIZE = 20000  # REDUCED from 50000 (faster tokenizer training)
BPE_MODEL_PREFIX = str(TOKENIZER_DIR / "legal_bpe_fast")
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
# Model Architecture Config (SMALLER MODEL)
# ========================
# Encoder
EMBEDDING_DIM = 128  # REDUCED from 256
ENCODER_HIDDEN_DIM = 256  # REDUCED from 512
ENCODER_NUM_LAYERS = 1  # REDUCED from 2
ENCODER_DROPOUT = 0.2  # REDUCED from 0.3
ENCODER_BIDIRECTIONAL = True

# Decoder
DECODER_HIDDEN_DIM = 256  # REDUCED from 512
DECODER_NUM_LAYERS = 1
DECODER_DROPOUT = 0.2

# Attention
ATTENTION_DIM = 256  # REDUCED from 512

# Sequence limits (SHORTER SEQUENCES)
MAX_ENCODER_LEN = 256  # REDUCED from 512
MAX_DECODER_LEN = 100  # REDUCED from 150

# ========================
# Training Config (FASTER)
# ========================
BATCH_SIZE = 16  # INCREASED from 8 (fewer iterations)
NUM_EPOCHS = 10  # REDUCED from 20
LEARNING_RATE = 0.0015  # Slightly higher for faster convergence
GRAD_CLIP = 5.0

# Loss weights
COVERAGE_LOSS_WEIGHT = 1.0
USE_COVERAGE = True

# Validation
VAL_SPLIT = 0.1
EVAL_EVERY_N_STEPS = 200  # REDUCED from 500 (more frequent validation)
SAVE_EVERY_N_EPOCHS = 2  # Save every 2 epochs instead of 1

# Early stopping
PATIENCE = 3

# ========================
# Device Config
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0

# ========================
# Logging Config
# ========================
LOG_INTERVAL = 50  # REDUCED from 100 (more frequent logs)
VERBOSE = True

# ========================
# Inference Config
# ========================
BEAM_SIZE = 4
MIN_DECODE_LEN = 20  # REDUCED from 35
MAX_DECODE_LEN = 80  # REDUCED from 120

# ========================
# Reproducibility
# ========================
RANDOM_SEED = 42

print(f"FAST Configuration loaded. Device: {DEVICE}")
print(f"Model size: ~25M parameters (vs 117M in full config)")
print(f"Dataset: {SAMPLE_FRACTION*100:.0f}% sample ({int(7030*SAMPLE_FRACTION)} examples)")
print(f"Estimated training time: 1-2 hours on CPU")
print(f"Output directory: {OUTPUT_DIR}")
