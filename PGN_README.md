# Hybrid Extractive-Abstractive Legal Text Summarization

## Overview

This repository implements a complete **hybrid extractive-abstractive summarization pipeline** for legal documents using PyTorch. The system combines:

1. **Extractive Summarization**: Uses the existing `SentenceSummarizer` module to select salient sentences from input documents
2. **Abstractive Summarization**: Implements a **Pointer-Generator Network (PGN)** with coverage mechanism to generate human-like summaries

## Architecture

### Pipeline Flow

```
Legal Document
    ↓
[Sentence Segmentation]
    ↓
[Extractive Filtering] ← SentenceSummarizer (TextRank + TF-IDF + Position + CNN)
    ↓
Top-K Salient Sentences
    ↓
[SentencePiece Tokenization] ← BPE Tokenizer
    ↓
[Pointer-Generator Network]
    ├─ BiLSTM Encoder
    ├─ LSTM Decoder
    ├─ Bahdanau Attention
    ├─ Copy Mechanism
    └─ Coverage Loss
    ↓
Generated Summary
```

### Key Components

#### 1. Extractive Component
- **Module**: `src/summarizer.py` (existing `SentenceSummarizer`)
- **Features**:
  - TextRank graph-based scoring
  - TF-IDF cosine similarity
  - Position-based scoring
  - Optional CNN boundary probabilities
  - Weighted combination of features

#### 2. Abstractive Component (Pointer-Generator Network)
- **Encoder**: Bidirectional LSTM
  - Processes extractively-filtered sentences
  - Produces contextualized representations
  - Hidden states reduced for decoder compatibility

- **Decoder**: LSTM with Bahdanau Attention
  - Generates summary one token at a time
  - Attends to encoder outputs
  - Maintains coverage vector

- **Pointer-Generator Mechanism**:
  - Computes generation probability `p_gen`
  - Can either generate from vocabulary or copy from source
  - Handles OOV (out-of-vocabulary) words effectively

- **Coverage Mechanism**:
  - Tracks cumulative attention to prevent repetition
  - Coverage loss penalizes re-attending to same positions

## Project Structure

```
LeXI-Phase-2/
├── config.py                  # Configuration and hyperparameters
├── train.py                   # Main training script
├── inference.py               # Inference and evaluation script
├── prepare_data.py            # Data preparation for tokenizer
├── utils.py                   # Training utilities (loss, metrics, checkpointing)
│
├── models/
│   ├── __init__.py
│   ├── encoder.py             # BiLSTM encoder
│   ├── decoder.py             # LSTM decoder with attention
│   ├── attention.py           # Bahdanau attention mechanism
│   └── pointer_generator.py  # Complete PGN model
│
├── data_utils/
│   ├── __init__.py
│   ├── preprocessing.py       # Sentence segmentation, extractive filtering, tokenization
│   └── dataset.py             # PyTorch Dataset and DataLoader
│
├── src/
│   └── summarizer.py          # Existing extractive summarizer
│
├── summariser_dataset/
│   ├── train.csv              # Training data (Text, Summary columns)
│   └── test.csv               # Test data (Text, Summary columns)
│
└── pgn_output/                # Output directory (created automatically)
    ├── models/                # Model checkpoints
    ├── tokenizer/             # SentencePiece tokenizer files
    ├── logs/                  # Training logs
    └── results/               # Generated summaries
```

## Requirements

### Python Dependencies

```bash
pip install torch numpy pandas tqdm sentencepiece scikit-learn nltk
```

### Required Packages
- **PyTorch** (>=1.9.0): Deep learning framework
- **SentencePiece** (>=0.1.96): BPE tokenization
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **tqdm**: Progress bars
- **scikit-learn**: Used by extractive summarizer
- **NLTK**: Used by extractive summarizer

## Dataset Format

The pipeline expects CSV files with two columns:

- **Text**: Full legal document text
- **Summary**: Gold human-written summary

Example:
```csv
Text,Summary
"Appeal No. LXVI of 1949. Appeal from the High Court...","The charge created in respect of municipal property..."
```

## Usage

### 1. Training

#### Step 1: Prepare Data and Train Tokenizer

The training script automatically handles tokenizer training if not found. Alternatively, prepare the data explicitly:

```bash
python prepare_data.py
```

This script:
- Loads training documents
- Applies extractive filtering to each document
- Creates a corpus file for tokenizer training
- Combines source texts and summaries

#### Step 2: Train the Model

```bash
python train.py
```

**Training Process**:
1. Loads/trains SentencePiece tokenizer (50K vocab, BPE)
2. Creates datasets with extractive filtering
3. Splits into train/validation sets (90%/10%)
4. Initializes Pointer-Generator Network
5. Trains with Adam optimizer
6. Validates after each epoch
7. Saves best checkpoints
8. Logs metrics to `pgn_output/logs/training_log.jsonl`

**Key Hyperparameters** (in `config.py`):
- `BATCH_SIZE = 8`: Batch size for training
- `NUM_EPOCHS = 20`: Maximum training epochs
- `LEARNING_RATE = 0.001`: Adam learning rate
- `EXTRACTIVE_TOP_K = 10`: Number of sentences to extract
- `MAX_ENCODER_LEN = 512`: Maximum input sequence length
- `MAX_DECODER_LEN = 150`: Maximum output sequence length
- `COVERAGE_LOSS_WEIGHT = 1.0`: Weight for coverage loss

### 2. Inference

#### Generate Summary for Single Document

```bash
python inference.py \
    --checkpoint pgn_output/models/checkpoint_epoch10_step5000.pt \
    --text "Your legal document text here..."
```

#### Generate Summaries for Test Set

```bash
python inference.py \
    --checkpoint pgn_output/models/checkpoint_epoch10_step5000.pt \
    --input summariser_dataset/test.csv \
    --output pgn_output/results/test_predictions.csv
```

#### Options:
- `--checkpoint`: Path to trained model checkpoint (required)
- `--input`: Input CSV file for batch processing
- `--output`: Output CSV file path
- `--text`: Direct text input for single document
- `--max-length`: Maximum summary length (default: 120)
- `--no-extractive`: Skip extractive filtering step

### 3. Configuration

Edit `config.py` to customize:

**Model Architecture**:
```python
EMBEDDING_DIM = 256
ENCODER_HIDDEN_DIM = 512
DECODER_HIDDEN_DIM = 512
ATTENTION_DIM = 512
```

**Training Settings**:
```python
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
GRAD_CLIP = 5.0
```

**Extractive Filtering**:
```python
EXTRACTIVE_TOP_K = 10  # Number of sentences to extract
```

**Coverage Mechanism**:
```python
USE_COVERAGE = True
COVERAGE_LOSS_WEIGHT = 1.0
```

## Model Details

### Loss Function

The total loss combines two components:

```
L_total = L_NLL + λ * L_coverage

where:
- L_NLL: Negative log-likelihood of target tokens
- L_coverage: Σ_t Σ_i min(attention_t_i, coverage_t_i)
- λ: Coverage loss weight (default: 1.0)
```

### Training Process

1. **Extractive Filtering**: Each document is processed by `SentenceSummarizer` to select top-K salient sentences
2. **Encoding**: Extracted sentences are concatenated and tokenized using SentencePiece (BPE)
3. **Forward Pass**: 
   - Encoder processes the extractively-filtered text
   - Decoder generates summary tokens autoregressively
   - Attention mechanism focuses on relevant encoder positions
   - Coverage tracks attention history
4. **Loss Computation**: NLL + coverage loss
5. **Backward Pass**: Gradient descent with clipping
6. **Validation**: Model evaluated on held-out validation set

### OOV Handling

The pointer-generator mechanism handles out-of-vocabulary (OOV) words:

1. **Extended Vocabulary**: Creates temporary IDs for OOV words in each example
2. **Copy Mechanism**: Allows copying OOV words directly from source
3. **Generation Probability**: `p_gen` controls mix of generation vs. copying

## Key Features

✅ **Hybrid Approach**: Combines extractive and abstractive methods  
✅ **Copy Mechanism**: Handles OOV words and preserves factual accuracy  
✅ **Coverage Loss**: Reduces repetition in generated summaries  
✅ **Attention Visualization**: Attention weights tracked for interpretability  
✅ **Batched Processing**: Efficient GPU utilization with batching  
✅ **Checkpointing**: Automatic model saving and recovery  
✅ **Logging**: Comprehensive training metrics logged  
✅ **Research-Grade**: Clean, documented, modular code

## Expected Results

### Training Metrics
- **Training Loss**: Should decrease steadily over epochs
- **Validation Loss**: Monitor for overfitting
- **Coverage Loss**: Should decrease as model learns to avoid repetition

### Output Quality
- **Abstractive**: Summaries should be abstractive, not just extractive
- **Coherent**: Well-formed sentences with proper grammar
- **Factual**: Key information from source preserved (via copy mechanism)
- **Concise**: Respects maximum length constraints

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `MAX_ENCODER_LEN` or `MAX_DECODER_LEN`
- Use gradient accumulation

**2. Slow Training**
- Ensure GPU is being used (check `DEVICE` in output)
- Reduce `EXTRACTIVE_TOP_K` to process shorter sequences
- Increase `BATCH_SIZE` if memory allows

**3. Poor Summary Quality**
- Train for more epochs
- Increase `COVERAGE_LOSS_WEIGHT` if summaries are repetitive
- Adjust `EXTRACTIVE_TOP_K` to include more/fewer sentences
- Check extractive summarizer weights in `data_utils/preprocessing.py`

**4. Tokenizer Errors**
- Delete `pgn_output/tokenizer/` and retrain
- Check corpus file has sufficient data

## Citation

This implementation is based on:

```bibtex
@article{see2017get,
  title={Get To The Point: Summarization with Pointer-Generator Networks},
  author={See, Abigail and Liu, Peter J and Manning, Christopher D},
  journal={arXiv preprint arXiv:1704.04368},
  year={2017}
}
```

## License

This code is for research purposes. Please cite appropriately if used in publications.

## Contact

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This is a research implementation designed for legal text summarization. The code prioritizes clarity and modularity for research reproducibility.
