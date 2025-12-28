# 📊 Evaluation Metrics Guide

## After Training: What We Evaluate

Once your model is trained, we evaluate it comprehensively using multiple metrics to assess different aspects of summary quality.

---

## 📋 Standard Summarization Metrics

### 1. **ROUGE Scores** (Primary Metric) ⭐

**What it measures**: Overlap between generated and reference summaries

**Variants:**
- **ROUGE-1**: Unigram (single word) overlap
- **ROUGE-2**: Bigram (two-word phrase) overlap  
- **ROUGE-L**: Longest common subsequence

**For each variant, we report:**
- Precision: What % of generated words are correct
- Recall: What % of reference words were captured
- **F1-Score**: Harmonic mean (main metric) ⭐

**Interpretation:**
```
ROUGE-1 F1:  0.45  →  Good
ROUGE-2 F1:  0.20  →  Good  
ROUGE-L F1:  0.40  →  Good
```

**What's good:**
- ROUGE-1 > 0.40: Excellent
- ROUGE-2 > 0.18: Excellent
- ROUGE-L > 0.35: Excellent

---

### 2. **BLEU Score**

**What it measures**: Precision-based n-gram matching (1-4 grams)

**Range**: 0.0 to 1.0 (higher is better)

**Interpretation:**
```
BLEU: 0.30  →  Good quality
BLEU: 0.40  →  Very good
BLEU: 0.50+ →  Excellent (rare in summarization)
```

**Note**: BLEU was designed for translation, but widely used in summarization

---

### 3. **METEOR Score**

**What it measures**: Semantic similarity including:
- Exact word matches
- Stem matches (running → run)
- Synonym matches (via WordNet)

**Range**: 0.0 to 1.0 (higher is better)

**Interpretation:**
```
METEOR: 0.25  →  Decent
METEOR: 0.30  →  Good
METEOR: 0.35+ →  Very good
```

**Advantage**: More sophisticated than BLEU, considers meaning

---

### 4. **BERTScore**

**What it measures**: Contextual semantic similarity using BERT embeddings

**Components:**
- Precision: Generated words match reference semantically
- Recall: Reference words captured semantically
- F1: Harmonic mean

**Range**: Typically 0.85-0.95 for good summaries

**Interpretation:**
```
BERTScore F1: 0.88  →  Good semantic match
BERTScore F1: 0.90+ →  Excellent semantic match
```

**Advantage**: Captures meaning beyond exact word matches

---

## 🔧 Custom Metrics

### 5. **Length Metrics**

**What we measure:**

```
Generated Length:     Average words in your summaries
Reference Length:     Average words in gold summaries
Compression Ratio:    Generated / Source (how much compression)
Length Ratio:         Generated / Reference (vs. gold standard)
```

**Ideal values:**
```
Compression: 0.05-0.15  (5-15% of original)
Length Ratio: 0.8-1.2   (Similar to reference)
```

---

### 6. **Abstractiveness Score**

**What it measures**: How "abstractive" vs "extractive" the summary is

**Components:**
- **Novel Unigrams**: % of words not in source (new words used)
- **Novel Bigrams**: % of 2-word phrases not in source
- **Abstractiveness**: Average of above

**Range**: 0.0 to 1.0

**Interpretation:**
```
Abstractiveness: 0.2-0.3  →  Mostly extractive
Abstractiveness: 0.4-0.6  →  Good mix (IDEAL)
Abstractiveness: 0.7+     →  Highly abstractive
```

**Why it matters**: Shows if model truly generates new text vs. copy-paste

---

## 📊 Example Evaluation Output

When you run `python evaluate.py`, you'll see:

```
================================================================================
EVALUATION RESULTS
================================================================================

📊 ROUGE Scores (overlap-based):
  ROUGE-1: 0.4523 (±0.1234)
  ROUGE-2: 0.2145 (±0.0987)
  ROUGE-L: 0.4012 (±0.1156)

📊 Other Metrics:
  BLEU: 0.3456 (±0.1023)
  METEOR: 0.3145 (±0.0876)
  BERTScore: 0.8934 (±0.0234)

📊 Length Metrics:
  Avg generated length: 82.3 words
  Avg reference length: 87.5 words
  Compression ratio: 0.089
  Length ratio (gen/ref): 0.941

📊 Abstractiveness:
  Novel unigrams: 0.453
  Novel bigrams: 0.612
  Abstractiveness: 0.532

✓ Detailed results saved to: pgn_output/results/evaluation_results.csv
✓ Aggregated metrics saved to: pgn_output/results/evaluation_metrics.json
```

---

## 🎯 How to Run Evaluation

### After training is complete:

```powershell
# Evaluate on test set
python evaluate.py \
    --checkpoint pgn_output_fast/models/checkpoint_epoch10_step660.pt \
    --input summariser_dataset/test.csv \
    --output evaluation_results.csv
```

### Quick test (first 50 examples):

```powershell
python evaluate.py \
    --checkpoint [model_path] \
    --input summariser_dataset/test.csv \
    --max-examples 50
```

---

## 📦 Required Packages for Evaluation

```powershell
# Install evaluation libraries
pip install rouge-score nltk bert-score
```

**What each does:**
- `rouge-score`: ROUGE metrics
- `nltk`: BLEU, METEOR
- `bert-score`: BERTScore (semantic similarity)

---

## 📈 What Makes a Good Summary (Target Metrics)

Based on legal summarization research:

| Metric | Target | Explanation |
|--------|--------|-------------|
| **ROUGE-1** | > 0.40 | Good word overlap |
| **ROUGE-2** | > 0.18 | Good phrase overlap |
| **ROUGE-L** | > 0.35 | Good sequence match |
| **BLEU** | > 0.30 | Good precision |
| **METEOR** | > 0.28 | Good semantic match |
| **BERTScore** | > 0.88 | Good contextual match |
| **Abstractiveness** | 0.4-0.6 | Good mix of abstractive/extractive |
| **Length Ratio** | 0.8-1.2 | Appropriate length |

---

## 🔍 Qualitative Analysis

Beyond automatic metrics, you should also check:

### 1. **Factual Accuracy**
- Do summaries contain correct information?
- No hallucinations (making up facts)?

### 2. **Fluency**
- Are summaries grammatically correct?
- Do they read naturally?

### 3. **Coherence**
- Do sentences flow logically?
- Is the summary well-structured?

### 4. **Coverage**
- Are key points from source included?
- Nothing important missing?

### 5. **Conciseness**
- No unnecessary details?
- Gets to the point?

---

## 📊 Interpretation Guide

### Scenario 1: High ROUGE, Low Abstractiveness
```
ROUGE-1: 0.50
Abstractiveness: 0.20
```
**Interpretation**: Model is mostly copying from source (extractive behavior)  
**Action**: Model working but not very abstractive

---

### Scenario 2: Low ROUGE, High Abstractiveness
```
ROUGE-1: 0.25
Abstractiveness: 0.70
```
**Interpretation**: Model generating new text but not matching references  
**Action**: Possible hallucination or poor quality

---

### Scenario 3: Balanced (IDEAL)
```
ROUGE-1: 0.45
ROUGE-2: 0.22
Abstractiveness: 0.50
BERTScore: 0.90
```
**Interpretation**: Good abstractive summarization with factual accuracy  
**Action**: Model performing well! ✅

---

## 📁 Output Files

After evaluation, you get:

1. **`evaluation_results.csv`**: 
   - Each test example with its metrics
   - Source, reference, generated summary
   - All individual scores

2. **`evaluation_metrics.json`**:
   - Aggregated statistics
   - Mean and std for each metric
   - Easy to parse for analysis

---

## 🎓 Research Context

These metrics are standard in summarization research:

- **ROUGE**: Most widely used (See et al., 2017; Nallapati et al., 2016)
- **BLEU/METEOR**: Borrowed from machine translation
- **BERTScore**: Modern semantic evaluation (Zhang et al., 2020)
- **Abstractiveness**: Measures generation vs. extraction

**Your paper should report:**
- All ROUGE scores (1, 2, L)
- At least one semantic metric (METEOR or BERTScore)
- Abstractiveness score
- Qualitative analysis of 50-100 examples

---

## ✅ After Tomorrow's Training

1. **Run evaluation**: `python evaluate.py --checkpoint [model] --input test.csv`
2. **Check ROUGE-1 > 0.35**: If yes, model is working ✅
3. **Review sample outputs**: Read 10-20 summaries manually
4. **Compare to baselines**: How does it compare to your extractive-only approach?

---

**Tomorrow when you're ready, the complete workflow will be:**

```powershell
# 1. Train (1-2 hours)
python train_fast.py

# 2. Generate summaries
python inference.py --checkpoint [best_model] --input test.csv --output summaries.csv

# 3. Evaluate
python evaluate.py --checkpoint [best_model] --input test.csv
```

Good luck tomorrow! 🚀
