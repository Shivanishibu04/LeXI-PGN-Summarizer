# Summarizer Design Documentation

## Overview

The LeXIDesk summarization module provides extractive summarization with sentence-weight attribution. It combines multiple scoring components to assign importance scores to sentences, then selects the top-k sentences or a percentage based on compression ratio.

## Architecture

### Core Components

1. **Sentence Segmentation**: Uses the existing CNN-CRF pipeline to split text into sentences
2. **Weight Computation**: Multi-component scoring system
3. **Sentence Selection**: Top-k or compression-based selection
4. **Output Formatting**: Displays selected sentences with their contribution percentages

## Scoring Components

### 1. CNN Probability Score (Weight: 25% default)

**Purpose**: Leverage CNN boundary detection confidence as a signal for sentence importance.

**Implementation**:
- If hybrid model is used, CNN probabilities are computed for delimiter tokens
- These probabilities indicate how confident the model is that a token is a sentence boundary
- Higher confidence boundaries may indicate more important sentence breaks
- Probabilities are normalized per sentence

**Fallback**: If CNN model is not available or probabilities are missing, this component contributes 0.0 (weights are renormalized).

### 2. TextRank Score (Weight: 35% default)

**Purpose**: Graph-based centrality measure to identify sentences central to the document.

**Implementation**:
- Builds a similarity graph where nodes are sentences
- Edge weights are Jaccard similarity between sentence token sets
- Uses PageRank algorithm to compute centrality scores
- Sentences with high centrality are more connected to other sentences

**Algorithm**:
```
1. Tokenize each sentence (remove stopwords, lowercase)
2. For each sentence pair (i, j):
   - Compute Jaccard similarity: |tokens_i ∩ tokens_j| / |tokens_i ∪ tokens_j|
   - If similarity > threshold (0.1), add edge with weight = similarity
3. Run PageRank on the graph
4. Normalize scores to sum to 1
```

**Dependencies**: Requires `networkx`. Falls back to uniform scores if not available.

### 3. TF-IDF / Embedding Score (Weight: 30% default)

**Purpose**: Measure how similar each sentence is to the document centroid (average representation).

**Implementation**:

**TF-IDF Mode** (default):
- Compute TF-IDF vectors for all sentences
- Compute document centroid (mean of all sentence vectors)
- Compute cosine similarity between each sentence and centroid
- Sentences similar to the centroid are more representative of the document

**Embedding Mode** (if `sentence-transformers` available):
- Use pre-trained sentence embeddings (default: `all-MiniLM-L6-v2`)
- Compute document centroid in embedding space
- Compute cosine similarity
- Generally provides better semantic similarity than TF-IDF

**Dependencies**: 
- TF-IDF: Requires `scikit-learn`
- Embeddings: Requires `sentence-transformers` (optional)

### 4. Position Score (Weight: 10% default)

**Purpose**: Favor earlier sentences (common in legal documents where key information appears first).

**Implementation**:
- Exponential decay: `score[i] = exp(-i * decay_rate)`
- Decay rate: 0.1 (configurable)
- First sentence gets highest score
- Normalized to sum to 1

## Weight Combination

Final sentence weight is computed as:

```
weight[i] = (
    cnn_weight * cnn_score[i] +
    textrank_weight * textrank_score[i] +
    tfidf_weight * similarity_score[i] +
    position_weight * position_score[i]
)
```

Weights are then normalized to sum to 1:

```
weight[i] = weight[i] / sum(weights)
```

## Sentence Selection

### Top-K Selection

Selects the k sentences with highest weights:

```python
selected_indices = argsort(weights)[::-1][:k]
```

### Compression Ratio

Selects a percentage of sentences:

```python
n_select = max(1, int(len(sentences) * compression))
selected_indices = argsort(weights)[::-1][:n_select]
```

### Preserve Order

If `preserve_order=True`, selected sentences are sorted by original position:

```python
selected_indices = sorted(selected_indices)  # Maintain original order
```

## Integration with Segmentation Pipeline

The summarizer integrates with the existing segmentation pipeline:

1. **Input**: Raw text string
2. **Segmentation**: `segment_text()` splits text into sentences using CRF model
3. **CNN Probabilities**: If hybrid model is used, CNN probabilities are extracted for boundaries
4. **Summarization**: `SentenceSummarizer.summarize()` computes weights and selects sentences
5. **Output**: Selected sentences with their weights and percentages

## Fallback Behavior

The system gracefully degrades when optional dependencies are missing:

| Component | Required Library | Fallback Behavior |
|-----------|------------------|-------------------|
| TextRank | networkx | Uniform scores |
| TF-IDF | scikit-learn | Uniform scores |
| Embeddings | sentence-transformers | Falls back to TF-IDF |
| CNN Probs | Trained CNN model | Zero contribution (weights renormalized) |
| NLTK | nltk | Simple regex tokenization |

## Configuration

### Default Weights

```python
cnn_prob_weight = 0.25
textrank_weight = 0.35
tfidf_weight = 0.30
position_weight = 0.10
```

### Custom Weights

You can customize weights when initializing:

```python
summarizer = SentenceSummarizer(
    cnn_prob_weight=0.1,
    textrank_weight=0.5,
    tfidf_weight=0.3,
    position_weight=0.1
)
```

Weights are automatically normalized to sum to 1.

## Extending to Abstractive Summarization

To extend the system to abstractive summarization, consider:

### 1. RAG (Retrieval-Augmented Generation) Pipeline

**Components**:
- **Chunking**: Split documents into overlapping chunks
- **Retrieval**: BM25 or dense retrieval (e.g., FAISS) to find relevant chunks
- **Generation**: Use a generative model (GPT, BART, T5) to generate summary
- **Evaluation**: ROUGE, BERTScore metrics

**Implementation Sketch**:

```python
# src/abstractive.py (skeleton)
class AbstractiveSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.retriever = BM25Retriever()  # or DenseRetriever
    
    def summarize(self, text, max_length=150):
        # 1. Chunk document
        chunks = self.chunk_text(text)
        
        # 2. Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve(chunks, top_k=5)
        
        # 3. Generate summary
        input_text = " ".join(relevant_chunks)
        summary = self.generate(input_text, max_length)
        
        return summary
```

### 2. Fine-Tuning for Legal Domain

**Requirements**:
- GPU with ≥16GB VRAM (for BART-large) or ≥40GB (for larger models)
- Legal summarization dataset (e.g., Legal-BERT dataset, custom annotations)
- Training framework (HuggingFace Transformers, PEFT/LoRA for efficiency)

**Steps**:
1. Prepare dataset: (document, summary) pairs
2. Fine-tune model on legal domain data
3. Evaluate on held-out test set
4. Deploy with quantization if needed

### 3. Evaluation Metrics

- **ROUGE**: Overlap-based metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore**: Semantic similarity using BERT embeddings
- **BLEU**: N-gram overlap (less suitable for summarization)
- **METEOR**: Considers synonyms and paraphrases

## Performance Considerations

### Computational Complexity

- **TextRank**: O(n²) for graph construction, O(n) for PageRank
- **TF-IDF**: O(n * m) where n = sentences, m = vocabulary size
- **Embeddings**: O(n * d) where d = embedding dimension (384 for MiniLM)
- **Overall**: Linear in number of sentences for most components

### Memory Usage

- **TF-IDF**: Stores sparse matrix (typically <100MB for 1000 sentences)
- **Embeddings**: ~1.5MB per 1000 sentences (384-dim float32)
- **TextRank Graph**: O(n²) edges in worst case, but typically sparse

### Optimization Tips

1. **Batch Processing**: Process multiple documents in batches
2. **Caching**: Cache embeddings for repeated documents
3. **Early Stopping**: For very long documents, consider chunking first
4. **Model Quantization**: Use quantized models for embeddings if memory-constrained

## Future Enhancements

1. **Diversity Penalty**: Add MMR (Maximal Marginal Relevance) to avoid redundant sentences
2. **Query-Focused Summarization**: Allow user queries to guide selection
3. **Multi-Document Summarization**: Extend to summarize multiple related documents
4. **Abstractive Pipeline**: Implement RAG-based abstractive summarization
5. **Domain Adaptation**: Fine-tune on legal-specific datasets
6. **Interactive Summarization**: Allow users to adjust weights and see results in real-time

## References

- TextRank: [Mihalcea & Tarau, 2004] "TextRank: Bringing Order into Text"
- PageRank: [Page et al., 1999] "The PageRank Citation Ranking"
- Extractive Summarization: [Nallapati et al., 2017] "SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization"
- Abstractive Summarization: [Lewis et al., 2019] "BART: Denoising Sequence-to-Sequence Pre-training"


