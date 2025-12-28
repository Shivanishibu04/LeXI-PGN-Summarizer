# Quick Start Guide: Hybrid Summarization Pipeline

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements_pgn.txt
```

### Step 2: Validate Setup
```bash
python validate_setup.py
```

### Step 3: Train the Model
```bash
python train.py
```

---

## 📋 Complete Workflow

### 1. **Preparation**

Install all dependencies:
```bash
pip install torch pandas numpy tqdm sentencepiece scikit-learn nltk
```

Check dataset structure:
```bash
python -c "import pandas as pd; df = pd.read_csv('summariser_dataset/train.csv', nrows=2); print(df.columns)"
```

### 2. **Run Example (Optional)**

See extractive component in action:
```bash
python example.py
```

This demonstrates:
- Sentence segmentation
- Extractive filtering using SentenceSummarizer
- Top-K sentence selection with weights

### 3. **Training**

#### Automatic (Recommended)
```bash
python train.py
```

This automatically:
- Prepares training corpus
- Trains SentencePiece tokenizer (if not exists)
- Creates train/val split
- Trains Pointer-Generator Network
- Saves checkpoints to `pgn_output/models/`
- Logs metrics to `pgn_output/logs/`

#### Manual (Step-by-Step)
```bash
# Step 1: Prepare data for tokenizer
python prepare_data.py

# Step 2: Train (will use prepared data)
python train.py
```

### 4. **Inference**

#### Single Document
```bash
python inference.py \
    --checkpoint pgn_output/models/checkpoint_epoch10_step5000.pt \
    --text "Your legal document here..."
```

#### Batch Processing (Test Set)
```bash
python inference.py \
    --checkpoint pgn_output/models/checkpoint_epoch10_step5000.pt \
    --input summariser_dataset/test.csv \
    --output pgn_output/results/test_summaries.csv
```

---

## ⚙️ Configuration

### Key Settings in `config.py`:

```python
# Extractive Configuration
EXTRACTIVE_TOP_K = 10          # Number of sentences to extract

# Model Architecture
EMBEDDING_DIM = 256
ENCODER_HIDDEN_DIM = 512
DECODER_HIDDEN_DIM = 512

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Sequence Lengths
MAX_ENCODER_LEN = 512          # Maximum input length
MAX_DECODER_LEN = 150          # Maximum summary length

# Coverage
USE_COVERAGE = True
COVERAGE_LOSS_WEIGHT = 1.0
```

### Adjusting for Your Setup:

**Limited GPU Memory:**
```python
BATCH_SIZE = 4                 # Reduce batch size
MAX_ENCODER_LEN = 256          # Reduce sequence length
ENCODER_HIDDEN_DIM = 256       # Reduce model size
```

**Faster Training:**
```python
EXTRACTIVE_TOP_K = 5           # Extract fewer sentences
NUM_EPOCHS = 10                # Fewer epochs
```

**Better Quality:**
```python
EXTRACTIVE_TOP_K = 15          # More context
NUM_EPOCHS = 30                # More training
COVERAGE_LOSS_WEIGHT = 2.0     # Less repetition
```

---

## 📊 Monitoring Training

### View Training Progress

Training logs are saved to: `pgn_output/logs/training_log.jsonl`

```bash
# View last 5 epochs
tail -5 pgn_output/logs/training_log.jsonl
```

### Expected Output (Training)
```
Epoch 1/20
================================================================================
Training metrics:
  Total loss: 5.2341
  NLL loss: 4.8234
  Coverage loss: 0.4107
  Time: 15m 32s

Validation metrics:
  Total loss: 4.9821
  NLL loss: 4.6234
  Coverage loss: 0.3587

✓ New best validation loss: 4.9821
```

### Loss Trends
- **Total Loss**: Should decrease over epochs
- **NLL Loss**: Primary loss, should converge
- **Coverage Loss**: Should decrease (model learns to avoid repetition)

---

## 🔧 Troubleshooting

### Problem: Out of Memory

**Solution 1**: Reduce batch size
```python
# config.py
BATCH_SIZE = 4  # or even 2
```

**Solution 2**: Reduce sequence lengths
```python
# config.py
MAX_ENCODER_LEN = 256
MAX_DECODER_LEN = 100
```

### Problem: Training Too Slow

**Solution 1**: Check GPU usage
```python
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution 2**: Reduce extractive top-k
```python
# config.py
EXTRACTIVE_TOP_K = 5  # Process shorter sequences
```

### Problem: Poor Summary Quality

**Solution 1**: Train longer
```python
# config.py
NUM_EPOCHS = 30
```

**Solution 2**: Increase coverage weight (if repetitive)
```python
# config.py
COVERAGE_LOSS_WEIGHT = 2.0
```

**Solution 3**: Adjust extractive filtering
```python
# config.py
EXTRACTIVE_TOP_K = 12  # More context
```

### Problem: Tokenizer Errors

**Solution**: Delete and retrain
```bash
rm -rf pgn_output/tokenizer/*
python train.py  # Will retrain tokenizer
```

---

## 📁 Output Files

After training, you'll have:

```
pgn_output/
├── models/
│   ├── checkpoint_epoch1_step500.pt
│   ├── checkpoint_epoch2_step1000.pt
│   └── ...
│
├── tokenizer/
│   ├── legal_bpe.model           # Trained tokenizer
│   ├── legal_bpe.vocab           # Vocabulary
│   └── tokenizer_corpus.txt      # Training corpus
│
├── logs/
│   └── training_log.jsonl        # Training metrics
│
└── results/
    └── generated_summaries.csv   # Inference outputs
```

---

## 🎯 Expected Performance

### Training Time (Approximate)
- **Small dataset (<1000 examples)**: 30 min - 1 hour per epoch
- **Medium dataset (1000-10000)**: 1-3 hours per epoch
- **Large dataset (>10000)**: 3-6 hours per epoch

*Times are for GPU (NVIDIA RTX 2060 or similar). CPU training will be much slower.*

### Model Quality
- **Epoch 5-10**: Basic coherent summaries
- **Epoch 10-15**: Good quality, may have some repetition
- **Epoch 15-20**: High quality, factual, minimal repetition

---

## 💡 Tips for Best Results

1. **Data Quality**: Ensure your summaries are high quality
2. **Extractive Tuning**: Experiment with `EXTRACTIVE_TOP_K` (5-15)
3. **Coverage**: Use `COVERAGE_LOSS_WEIGHT=1.0` initially
4. **Validation**: Monitor validation loss to avoid overfitting
5. **Checkpoints**: Keep best 3 checkpoints (automatic)
6. **Early Stopping**: Patience of 3 epochs (automatic)

---

## 📚 Next Steps

1. ✅ **Validate Setup**: `python validate_setup.py`
2. ✅ **Run Example**: `python example.py`
3. ✅ **Train Model**: `python train.py`
4. ✅ **Generate Summaries**: `python inference.py --help`
5. ✅ **Evaluate Results**: Compare with gold summaries

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide
2. Read `PGN_README.md` for detailed documentation
3. Verify setup with `validate_setup.py`
4. Check configuration in `config.py`
5. Review training logs in `pgn_output/logs/`

---

**Happy Summarizing! 🎉**
