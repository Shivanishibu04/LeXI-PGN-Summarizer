# LeXI-Phase-2: Hybrid Legal Document Summarization

A hybrid extractive-abstractive summarization system for legal documents using Pointer-Generator Networks.

## 🎯 Overview

This project implements a complete pipeline for legal text summarization combining:
- **Extractive Component**: CNN-CRF sentence boundary detection + SentenceSummarizer
- **Abstractive Component**: Pointer-Generator Network with attention and coverage mechanism

## 🏗️ Architecture

```
Legal Document
    ↓
[CNN-CRF] Sentence Segmentation
    ↓
[SentenceSummarizer] Extract Top-K Sentences
    ↓
[SentencePiece BPE] Tokenization
    ↓
[Pointer-Generator Network]
  • BiLSTM Encoder
  • LSTM Decoder  
  • Bahdanau Attention
  • Copy Mechanism
  • Coverage Loss
    ↓
Abstractive Summary
```

## 📊 Features

- ✅ Hybrid extractive-abstractive approach
- ✅ CNN-CRF for accurate legal sentence boundary detection
- ✅ Pointer-Generator with copy mechanism for handling legal terminology
- ✅ Coverage mechanism to reduce repetition
- ✅ Comprehensive evaluation metrics (ROUGE, BLEU, METEOR, BERTScore)
- ✅ Fast training mode for CPU (1-2 hours)
- ✅ Research-grade code suitable for publication

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements_pgn.txt
pip install -r requirements_evaluation.txt
```

### Fast Training (1-2 hours on CPU)

```bash
python train_fast.py
```

### Generate Summaries

```bash
python inference.py \
    --checkpoint pgn_output_fast/models/checkpoint_epoch10_step660.pt \
    --input summariser_dataset/test.csv \
    --output results.csv
```

### Evaluate

```bash
python evaluate.py \
    --checkpoint pgn_output_fast/models/checkpoint_epoch10_step660.pt \
    --input summariser_dataset/test.csv
```

## 📁 Project Structure

```
LeXI-Phase-2/
├── models/                    # Neural network components
│   ├── encoder.py            # BiLSTM encoder
│   ├── decoder.py            # LSTM decoder with attention
│   ├── attention.py          # Bahdanau attention
│   └── pointer_generator.py # Complete PGN model
│
├── data_utils/               # Data processing
│   ├── preprocessing.py      # Extractive filtering & tokenization
│   └── dataset.py           # PyTorch dataset
│
├── src/                      # Existing models
│   ├── cnn_model.py         # CNN for sentence boundaries
│   ├── crf_model.py         # CRF model
│   └── summarizer.py        # Extractive summarizer
│
├── train.py                  # Full training script
├── train_fast.py            # Fast training (CPU optimized)
├── inference.py             # Generate summaries
├── evaluate.py              # Comprehensive evaluation
├── config.py                # Full configuration
└── config_fast.py           # Fast configuration
```

## 📊 Evaluation Metrics

The system evaluates generated summaries using:

- **ROUGE** (1, 2, L): Overlap-based metrics
- **BLEU**: Precision-based n-gram metric
- **METEOR**: Semantic similarity with synonyms
- **BERTScore**: Contextual embedding similarity
- **Abstractiveness**: Novel content generation
- **Length metrics**: Compression ratio, length ratio

## 🎯 Expected Performance

Fast training mode (1-2 hours, 15% data):
- ROUGE-1: 0.30-0.40
- ROUGE-2: 0.12-0.20
- Abstractiveness: 0.40-0.60

Full training mode (60-100 hours, 100% data):
- ROUGE-1: 0.40-0.48
- ROUGE-2: 0.18-0.26
- Production-quality summaries

## 📚 Documentation

- [`PGN_README.md`](PGN_README.md) - Detailed architecture and usage
- [`QUICK_START.md`](QUICK_START.md) - Quick start guide with commands
- [`TRAINING_TIME_GUIDE.md`](TRAINING_TIME_GUIDE.md) - Time estimates and optimization
- [`EVALUATION_METRICS.md`](EVALUATION_METRICS.md) - Metrics explanation
- [`INTEGRATION_CONFIRMED.md`](INTEGRATION_CONFIRMED.md) - Component integration details
- [`TOMORROW_CHECKLIST.md`](TOMORROW_CHECKLIST.md) - Step-by-step workflow

## 🛠️ Requirements

- Python 3.7+
- PyTorch 1.9+
- SentencePiece
- sklearn-crfsuite
- NLTK
- See `requirements_pgn.txt` and `requirements_evaluation.txt` for full list

## 🏆 Research Context

This implementation is based on:

```bibtex
@article{see2017get,
  title={Get To The Point: Summarization with Pointer-Generator Networks},
  author={See, Abigail and Liu, Peter J and Manning, Christopher D},
  journal={arXiv preprint arXiv:1704.04368},
  year={2017}
}
```

Extended for legal domain with hybrid extractive-abstractive approach.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{lexi-phase2,
  title={LeXI Phase 2: Hybrid Legal Document Summarization},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/[your-username]/LeXI-Phase-2}}
}
```

## 📄 License

[Your chosen license - e.g., MIT, Apache 2.0]

## 🙏 Acknowledgments

- Existing CNN-CRF sentence boundary detection system
- SentenceSummarizer extractive component
- Pointer-Generator Networks (See et al., 2017)

## 📧 Contact

[Your contact information or leave blank]

---

**Note**: Dataset files are not included in this repository due to size. The system expects CSV files in `summariser_dataset/` with columns: `Text` (full document) and `Summary` (gold summary).
