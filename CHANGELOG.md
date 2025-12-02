# Changelog

All notable changes to LeXIDesk will be documented in this file.

## [Unreleased]

### Added - Summarization Module

#### Features
- **Extractive Summarization**: New summarization module that generates extractive summaries with sentence-weight attribution
- **Multi-Component Scoring**: Combines CNN probabilities, TextRank, TF-IDF/embeddings, and position-based scores
- **CLI Integration**: Added `--summarize` flag to `predict.py` with support for:
  - File input (`--text-file`)
  - stdin input (`--stdin`)
  - Compression ratio (`--compression`)
  - Top-k selection (`--top-k`)
  - Order preservation (`--preserve-order`)
  - Enhanced embeddings (`--use-embeddings`)

#### New Files
- `src/summarizer.py`: Core summarization logic with sentence scoring and selection
- `tests/test_summarizer.py`: Comprehensive unit tests for summarization module
- `examples/summary_example.txt`: Example input/output documentation
- `docs/SUMMARIZER_DESIGN.md`: Detailed design documentation

#### Modified Files
- `predict.py`: 
  - Added CLI argument parsing with argparse
  - Integrated summarization functionality
  - Modified `segment_text()` to optionally return CNN probabilities
  - Added `main()` function for CLI entry point
  - Maintained backward compatibility with existing segmentation-only mode
- `requirements.txt`: 
  - Added `networkx` for TextRank implementation
  - Added `nltk` for tokenization
  - Added optional `sentence-transformers` (commented) for enhanced embeddings
- `README.md`: 
  - Added comprehensive documentation for summarization feature
  - Added usage examples and CLI options
  - Added troubleshooting section

#### Technical Details
- **Scoring Components**:
  - CNN probability score (25% default weight)
  - TextRank graph centrality (35% default weight)
  - TF-IDF/embedding similarity (30% default weight)
  - Position-based scoring (10% default weight)
- **Fallback Support**: Gracefully handles missing optional dependencies
- **Weight Normalization**: All scores normalized to sum to 1.0
- **Sentence Selection**: Supports both compression ratio and top-k modes

#### Testing
- Unit tests for weight normalization
- Tests for compression ratio and top-k selection
- Tests for order preservation
- Tests for edge cases (empty input, single sentence)
- Tests for CNN probability integration

#### Documentation
- Comprehensive README with usage examples
- Design documentation explaining scoring algorithms
- Example input/output file
- Notes on extending to abstractive summarization

### Changed
- `predict.py`: Refactored to support both segmentation-only and summarization modes
- Model loading: Made more robust with error handling and warnings

### Dependencies
- **New Required**: `networkx`, `nltk`
- **New Optional**: `sentence-transformers` (for enhanced embeddings)

## [Previous Versions]

[Add previous changelog entries here as needed]


