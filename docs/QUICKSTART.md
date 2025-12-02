# Quick Start Guide: Running LeXIDesk Summarization

This guide provides step-by-step instructions to run the summarization feature.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git (optional, if cloning the repository)

## Step 1: Navigate to Project Directory

Open a terminal/command prompt and navigate to the LeXIDesk project directory:

```bash
cd path/to/LexiDesk
```

For example:
```bash
cd C:\Users\sivap\OneDrive\Desktop\S7_Project_try\dev2\LexiDesk
```

## Step 2: Create and Activate Virtual Environment (Recommended)

### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### On Windows (Command Prompt):
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

### On Linux/Mac:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt when activated.

## Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- pandas, tqdm
- torch, tensorflow
- scikit-learn
- sklearn-crfsuite
- networkx (for TextRank)
- nltk (for tokenization)

### Optional: Install Enhanced Embeddings (Recommended for Better Quality)

For better summarization quality, install sentence-transformers:

```bash
pip install sentence-transformers
```

**Note**: This is optional but recommended. The system will work without it using TF-IDF.

## Step 4: Download NLTK Data (First Time Only)

If this is your first time using NLTK, download required data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

Or run Python interactively:
```bash
python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> exit()
```

## Step 5: Verify Models Exist

Check if trained models are available:

```bash
# On Windows (PowerShell)
dir saved_models

# On Linux/Mac
ls saved_models
```

You should see:
- `cnn_model.pth`
- `crf_baseline_model.joblib`
- `crf_hybrid_model.joblib`

### If Models Don't Exist: Train Them

If models are missing, train them first:

```bash
python -m src.crf_model
```

**Note**: This may take several minutes to complete. The script will:
1. Train the CNN model
2. Train the baseline CRF model
3. Train the hybrid CNN-CRF model
4. Save all models to `saved_models/`

## Step 6: Run Summarization

### Example 1: Basic Summarization (Uses Sample Text)

The simplest way to test summarization:

```bash
python predict.py --summarize
```

This will:
- Use the default sample legal text
- Segment it into sentences
- Compute sentence weights
- Display the summary with contribution percentages

### Example 2: Summarize from a Text File

Create a text file with your legal text:

```bash
# Create a sample file (Windows PowerShell)
@"
The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review. This principle is outlined in § 1.3(a) of the legal code. The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. All proceedings were documented by the F.B.I. for review.
"@ | Out-File -FilePath examples\sample.txt -Encoding utf8
```

Or manually create `examples/sample.txt` with your text.

Then run:

```bash
python predict.py --summarize --text-file examples/sample.txt
```

### Example 3: Summarize with Compression Ratio

Select top 20% of sentences:

```bash
python predict.py --summarize --text-file examples/sample.txt --compression 0.2
```

Select top 50% of sentences:

```bash
python predict.py --summarize --text-file examples/sample.txt --compression 0.5
```

### Example 4: Summarize with Top-K Sentences

Select top 3 sentences:

```bash
python predict.py --summarize --text-file examples/sample.txt --top-k 3
```

### Example 5: Read from stdin

Pipe text directly:

**On Windows (PowerShell):**
```powershell
Get-Content examples\sample.txt | python predict.py --summarize --stdin
```

**On Linux/Mac:**
```bash
cat examples/sample.txt | python predict.py --summarize --stdin
```

Or type directly:
```bash
python predict.py --summarize --stdin
# Then paste your text and press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows)
```

### Example 6: Use Enhanced Embeddings

If you installed sentence-transformers:

```bash
python predict.py --summarize --text-file examples/sample.txt --use-embeddings
```

## Step 7: Understanding the Output

The output will show:

1. **Original Text**: The input text
2. **Segmented Sentences**: Numbered list of detected sentences
3. **Weights**: Each sentence's contribution percentage
4. **Summary**: Selected sentences with their original indices and percentages

Example output:
```
==================================================
--- Summarization Mode ---

Original Text:
---
[Your text here]
---

Segmenting input...

Segmented Sentences:
[1] Sentence 1...
[2] Sentence 2...
[3] Sentence 3...

Computing sentence weights...

Weights (normalized):
[1] 0.364 (36.4%)
[2] 0.210 (21.0%)
[3] 0.280 (28.0%)

Summary (top 2 sentences):
1) [1] Sentence 1... — 36.4%
2) [3] Sentence 3... — 28.0%

==================================================
```

## Step 8: Run Tests (Optional)

Verify everything works correctly:

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
pytest tests/test_summarizer.py -v
```

Expected output: All tests should pass.

## Troubleshooting

### Issue: "Error: A model file was not found"

**Solution**: Train the models first:
```bash
python -m src.crf_model
```

### Issue: "ModuleNotFoundError: No module named 'networkx'"

**Solution**: Install missing dependency:
```bash
pip install networkx
```

### Issue: "NLTK data not found"

**Solution**: Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Issue: "Warning: networkx not installed. TextRank will be disabled"

**Solution**: This is a warning, not an error. The system will still work but with reduced functionality. To enable TextRank:
```bash
pip install networkx
```

### Issue: Models take too long to train

**Solution**: This is normal. Training can take 10-30 minutes depending on your hardware. Be patient and let it complete.

### Issue: "FileNotFoundError: data/processed/train_data.csv"

**Solution**: Make sure you're in the correct directory and that the data files exist. Check:
```bash
dir data\processed
```

## Advanced Usage

### Custom Weight Configuration

To customize scoring weights, modify `src/summarizer.py` or create a custom script:

```python
from src.summarizer import SentenceSummarizer

# Custom weights
summarizer = SentenceSummarizer(
    cnn_prob_weight=0.1,
    textrank_weight=0.5,
    tfidf_weight=0.3,
    position_weight=0.1
)
```

### Batch Processing

To process multiple files, create a simple script:

```python
import os
from predict import load_models, segment_text
from src.summarizer import SentenceSummarizer

# Load models once
baseline_crf_model, hybrid_crf_model, cnn_model, char_to_idx, device, vocab_size = load_models()
model = hybrid_crf_model if hybrid_crf_model else baseline_crf_model

# Process each file
for filename in os.listdir('input_files'):
    with open(f'input_files/{filename}', 'r') as f:
        text = f.read()
    
    sentences, cnn_probs = segment_text(text, model, use_hybrid_features=True, return_cnn_probs=True)
    summarizer = SentenceSummarizer()
    selected, weights, _ = summarizer.summarize(sentences, text, cnn_probs, top_k=3)
    
    print(f"\n{filename}:")
    for sent in selected:
        print(f"  - {sent}")
```

## Next Steps

- Read `docs/SUMMARIZER_DESIGN.md` for detailed algorithm explanations
- Check `README.md` for more examples
- Review `examples/summary_example.txt` for sample outputs
- Explore the code in `src/summarizer.py` to understand the implementation

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure models are trained
4. Check that you're in the correct directory
5. Review the troubleshooting section above


