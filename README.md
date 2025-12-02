# LeXIDesk: Legal Text Sentence Boundary Detection and Summarization

LeXIDesk is a tool for legal text processing that provides:
1. **Sentence Boundary Detection (SBD)**: Uses hybrid CNN-CRF models to accurately segment legal documents into sentences
2. **Extractive Summarization**: Generates summaries with sentence-weight attribution

## Features

- **Hybrid CNN-CRF Model**: Combines CNN character-level features with CRF sequence labeling for robust sentence segmentation
- **Extractive Summarization**: Multi-component scoring system that combines:
  - CNN boundary probabilities (if available)
  - TextRank graph centrality scores
  - TF-IDF cosine similarity
  - Position-based scoring
- **Flexible CLI**: Command-line interface supporting file input, stdin, and various summarization options
- **Graceful Fallbacks**: Works with minimal dependencies, with optional enhancements via sentence-transformers

## Quick Start

**For detailed step-by-step instructions, see [docs/QUICKSTART.md](docs/QUICKSTART.md)**

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Train Models (if not already done)

```bash
python -m src.crf_model
```

### 3. Run Summarization

```bash
# Basic usage (uses sample text)
python predict.py --summarize

# From file
python predict.py --summarize --text-file examples/sample.txt

# With compression ratio (20% of sentences)
python predict.py --summarize --text-file examples/sample.txt --compression 0.2

# Top 3 sentences
python predict.py --summarize --text-file examples/sample.txt --top-k 3
```

## Installation

### Quick Installation (Minimal Dependencies)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### Full Installation (With Enhanced Embeddings)

For better summarization quality, install the optional sentence-transformers package:

```bash
pip install sentence-transformers
```

## Usage

### Sentence Segmentation

Basic usage (uses sample text):

```bash
python predict.py
```

### Summarization

#### Basic Summarization

```bash
python predict.py --summarize
```

#### Summarize from File

```bash
python predict.py --summarize --text-file examples/sample.txt
```

#### Summarize with Compression Ratio

Select top 20% of sentences:

```bash
python predict.py --summarize --text-file examples/sample.txt --compression 0.2
```

#### Summarize with Top-K Sentences

Select top 3 sentences:

```bash
python predict.py --summarize --text-file examples/sample.txt --top-k 3
```

#### Read from stdin

```bash
echo "Your legal text here..." | python predict.py --summarize --stdin
```

#### Use Enhanced Embeddings (if installed)

```bash
python predict.py --summarize --text-file examples/sample.txt --use-embeddings
```

### CLI Options

```
--summarize          Enable summarization mode
--text-file PATH     Path to input text file
--stdin              Read input from stdin
--compression RATIO  Compression ratio (0-1), e.g., 0.2 means 20% of sentences
--top-k N            Number of top sentences to select
--preserve-order     Preserve original sentence order in summary (default: True)
--use-hybrid         Use hybrid CNN-CRF model (default: True)
--use-embeddings     Use sentence-transformers for better similarity
```

## Example Output

```
==================================================
--- Summarization Mode ---

Original Text:
---
The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review. This principle is outlined in § 1.3(a) of the legal code. The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. All proceedings were documented by the F.B.I. for review.
---

Segmenting input...

Segmented Sentences:
[1] The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review.
[2] This principle is outlined in § 1.3(a) of the legal code.
[3] The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001.
[4] All proceedings were documented by the F.B.I. for review.

Computing sentence weights (components: cnn_prob=0.25, textrank=0.35, tfidf=0.30, position=0.10)...

Weights (normalized):
[1] 0.364 (36.4%)
[2] 0.210 (21.0%)
[3] 0.280 (28.0%)
[4] 0.146 (14.6%)

Summary (top 2 sentences, compression=0.5):
1) [1] The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review. — 36.4%
2) [3] The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. — 28.0%

==================================================
```

## Project Structure

```
LexiDesk/
├── src/
│   ├── cnn_model.py          # CNN model for sentence boundary detection
│   ├── crf_model.py          # CRF model training and evaluation
│   ├── feature_extractor.py  # Feature extraction for CRF
│   ├── summarizer.py         # Summarization module (NEW)
│   └── predict.py            # Prediction utilities
├── data/
│   ├── raw/                  # Raw JSONL data files
│   └── processed/            # Processed CSV files
├── saved_models/             # Trained model files
├── tests/
│   └── test_summarizer.py   # Unit tests for summarization (NEW)
├── examples/
│   └── summary_example.txt   # Example input/output (NEW)
├── docs/
│   └── SUMMARIZER_DESIGN.md # Design documentation (NEW)
├── predict.py               # Main CLI entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Training Models

Before using the tool, you need to train the models:

```bash
python -m src.crf_model
```

This will:
1. Train the CNN model for boundary detection
2. Train the baseline CRF model
3. Train the hybrid CNN-CRF model
4. Save all models to `saved_models/`

## Testing

Run unit tests:

```bash
pytest tests/test_summarizer.py -v
```

## Dependencies

### Required
- pandas
- tqdm
- torch
- tensorflow
- scikit-learn
- sklearn-crfsuite
- networkx
- nltk

### Optional
- sentence-transformers (for enhanced embeddings)

## How Summarization Works

The summarizer uses a multi-component scoring system:

1. **CNN Probability Score** (25%): Uses CNN boundary detection probabilities if available
2. **TextRank Score** (35%): Graph-based centrality using sentence similarity
3. **TF-IDF/Embedding Score** (30%): Cosine similarity between sentences and document centroid
4. **Position Score** (10%): Exponential decay favoring earlier sentences

Scores are normalized to sum to 1, and sentences are selected based on combined weights. See `docs/SUMMARIZER_DESIGN.md` for detailed documentation.

## Troubleshooting

### Models Not Found

If you see "Error: A model file was not found", train the models first:

```bash
python -m src.crf_model
```

### Missing Dependencies

If you see warnings about missing libraries:

```bash
# For networkx (TextRank)
pip install networkx

# For nltk (tokenization)
pip install nltk

# For sentence-transformers (optional, better embeddings)
pip install sentence-transformers
```

### NLTK Data

If NLTK tokenization fails, download required data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Citation

If you use LeXIDesk in your research, please cite:

[Add citation information here]

