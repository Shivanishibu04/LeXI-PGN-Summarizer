"""
Data preparation script for training SentencePiece tokenizer.

This script prepares the training data by:
1. Loading the training corpus
2. Applying extractive filtering to all documents
3. Extracting both source texts and summaries
4. Creating a combined corpus for tokenizer training
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from data_utils.preprocessing import apply_extractive_filtering


def prepare_tokenizer_corpus():
    """
    Prepare corpus for SentencePiece training.
    
    Creates a text file with one sentence per line, combining:
    - Extractively filtered source documents
    - Gold summaries
    
    This ensures the tokenizer learns from both source and target distributions.
    """
    print("="*80)
    print("PREPARING TOKENIZER TRAINING CORPUS")
    print("="*80)
    
    # Load training data
    print(f"\nLoading training data from {config.TRAIN_CSV}...")
    df = pd.read_csv(config.TRAIN_CSV)
    
    # Drop missing values
    df = df.dropna(subset=['Text', 'Summary'])
    print(f"Loaded {len(df)} training examples")
    
    # Prepare output file
    corpus_file = config.TOKENIZER_DIR / "tokenizer_corpus.txt"
    print(f"\nWriting corpus to {corpus_file}...")
    
    with open(corpus_file, 'w', encoding='utf-8') as f:
        # Process each document
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
            full_text = str(row['Text'])
            summary = str(row['Summary'])
            
            # Apply extractive filtering
            extracted_text, _, _ = apply_extractive_filtering(
                text=full_text,
                top_k=config.EXTRACTIVE_TOP_K,
                compression=config.EXTRACTIVE_COMPRESSION
            )
            
            # Use original text if extraction fails
            if not extracted_text:
                extracted_text = full_text
            
            # Write extracted text (one line per document for simplicity)
            # For better sentence segmentation, you might want to split into sentences
            f.write(extracted_text.replace('\n', ' ').strip() + '\n')
            
            # Write summary
            f.write(summary.replace('\n', ' ').strip() + '\n')
    
    print(f"\nCorpus prepared: {corpus_file}")
    print(f"File size: {corpus_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return str(corpus_file)


if __name__ == "__main__":
    prepare_tokenizer_corpus()
    print("\nDone!")
