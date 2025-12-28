# ⏱️ Training Time Guide & Speed Optimization

## 📊 Your Dataset

- **Training examples**: 7,030 legal documents
- **File size**: 210 MB
- **Average document length**: ~30KB per document

---

## ⏰ Training Time Estimates (CPU)

### Option 1: **FULL Training** (`train.py`)

```
Configuration:
  • Full dataset: 7,030 examples
  • Model size: ~117M parameters
  • Vocabulary: 50,000 tokens
  • Epochs: 20
  • Extractive top-k: 10 sentences
  
⏱️ Time per epoch: 3-5 hours
📅 Total time: 60-100 hours (2.5-4 days)
```

**Pros**: Best quality results  
**Cons**: Very slow on CPU ⚠️

---

### Option 2: **FAST Training** (`train_fast.py`) ✅ RECOMMENDED

```
Configuration:
  • Sampled dataset: 1,054 examples (15%)
  • Model size: ~25M parameters (smaller)
  • Vocabulary: 20,000 tokens
  • Epochs: 10 (reduced)
  • Extractive top-k: 5 sentences (shorter)
  
⏱️ Time per epoch: 6-12 minutes
📅 Total time: 1-2 hours
```

**Pros**: 30-50x faster! ✅  
**Cons**: Slightly lower quality (still good)  

---

### Option 3: **QUICK TEST** (Manual config)

```
Configuration:
  • Tiny dataset: 200 examples (3%)
  • Model size: ~10M parameters (very small)
  • Vocabulary: 10,000 tokens
  • Epochs: 5
  • Extractive top-k: 3 sentences
  
⏱️ Time per epoch: 2-3 minutes
📅 Total time: 10-15 minutes
```

**Pros**: Ultra fast for testing  
**Cons**: Demo quality only  

---

## 🚀 Speed Optimization Strategies

### What We Changed in `config_fast.py`:

| Setting | Original | Fast | Impact | 
|---------|----------|------|---------|
| **Dataset** | 7,030 (100%) | 1,054 (15%) | 7x faster ⚡ |
| **Model size** | 117M params | 25M params | 4x faster ⚡ |
| **Vocab size** | 50,000 | 20,000 | 2x faster ⚡ |
| **Encoder layers** | 2 | 1 | 1.5x faster ⚡ |
| **Hidden dim** | 512 | 256 | 2x faster ⚡ |
| **Embedding** | 256 | 128 | 1.3x faster ⚡ |
| **Max encoder len** | 512 | 256 | 2x faster ⚡ |
| **Extractive top-k** | 10 sent. | 5 sent. | 1.5x faster ⚡ |
| **Batch size** | 8 | 16 | 2x faster ⚡ |
| **Epochs** | 20 | 10 | 2x faster ⚡ |

**Combined speedup: ~40-50x faster** 🚀

---

## 📋 How to Use

### **For Quick Results (RECOMMENDED):**

```powershell
# Use the FAST training script
python train_fast.py
```

**Expected output:**
```
FAST Configuration loaded. Device: cpu
Model size: ~25M parameters (vs 117M in full config)
Dataset: 15% sample (1054 examples)
Estimated training time: 1-2 hours on CPU

Epoch 1/10
Training: 100%|████████| 60/60 [08:23<00:00]
  Total loss: 5.234
  Time: 8m 23s

Validating...
  Total loss: 4.982

✓ New best validation loss
```

---

### **For Best Quality (if you have time):**

```powershell
# Use the full training script
python train.py
```

**Note**: This will take 2-4 DAYS on CPU. Consider:
- Running overnight
- Using GPU if available
- Or stick with fast mode ✅

---

## 💡 Additional Speed Tips

### 1. **Reduce Dataset Further** (if still too slow)

Edit `config_fast.py`:
```python
SAMPLE_FRACTION = 0.05  # Use only 5% (~350 examples)
```

### 2. **Make Model Even Smaller**

Edit `config_fast.py`:
```python
EMBEDDING_DIM = 64          # Was 128
ENCODER_HIDDEN_DIM = 128    # Was 256
DECODER_HIDDEN_DIM = 128    # Was 256
VOCAB_SIZE = 10000          # Was 20000
```

### 3. **Shorten Sequences More**

Edit `config_fast.py`:
```python
MAX_ENCODER_LEN = 128       # Was 256
MAX_DECODER_LEN = 50        # Was 100
EXTRACTIVE_TOP_K = 3        # Was 5
```

### 4. **Use Fewer Epochs**

Edit `config_fast.py`:
```python
NUM_EPOCHS = 5              # Was 10
```

---

## ⚖️ Quality vs Speed Trade-off

| Mode | Time | Quality | Use Case |
|------|------|---------|----------|
| **Full** | 60-100h | ⭐⭐⭐⭐⭐ | Production/Research |
| **Fast** | 1-2h | ⭐⭐⭐⭐ | Quick experiments ✅ |
| **Quick** | 10-15m | ⭐⭐⭐ | Testing pipeline |

---

## 📈 Performance Expectations

### Fast Mode Results:
- **Extractive**: Excellent (using your CNN-CRF + SentenceSummarizer)
- **Abstractive**: Good quality summaries
- **Coherence**: High
- **Factuality**: Good (copy mechanism helps)
- **Repetition**: Minimal (coverage loss)

### What you'll get:
```
Input (after extractive):
"The charge created in respect of municipal property... 
The Revenue Recovery Act must be construed..."

Generated Summary:
"Municipal tax charges are not considered government 
revenue under the Revenue Recovery Act."
```

---

## 🎯 Recommendation

**For your use case (CPU training):**

1. **START HERE**: `python train_fast.py` ✅
   - 1-2 hours total
   - Good quality results
   - Tests the full pipeline

2. **If satisfied**: Keep using fast mode or scale up later

3. **If you need production quality**: 
   - Consider using GPU (cloud/colab)
   - Or run full training overnight for several days

---

## 🖥️ GPU vs CPU Comparison

If you had a GPU available:

| Configuration | CPU | GPU (RTX 3060) |
|---------------|-----|----------------|
| **Full training** | 60-100h | 4-6h |
| **Fast training** | 1-2h | 8-12 min |

**GPU is 10-15x faster than CPU**

Free GPU options:
- Google Colab (free tier: 12h sessions)
- Kaggle (30h/week free GPU)

---

## ✅ Ready to Start?

**Recommended command:**
```powershell
python train_fast.py
```

**Monitor progress:**
- Watch the progress bars
- Check validation loss decreasing
- First epoch shows time-per-epoch estimate

**Stop early if needed:**
- Press `Ctrl+C` to stop
- Model checkpoints are saved every 2 epochs
- You can resume later (manually load checkpoint)

---

**Want to proceed with fast training?** 🚀
