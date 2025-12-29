"""
Summarization module for LeXIDesk.

Provides extractive summarization with sentence-weight attribution using:
- CNN boundary probabilities (if available)
- TextRank scores
- TF-IDF cosine similarity
- Position-based scoring
"""

import re
import warnings
from typing import List, Tuple, Dict, Optional
import numpy as np

# Optional imports with graceful fallback
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("networkx not installed. TextRank will be disabled. Install with: pip install networkx")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed. TF-IDF will be disabled. Install with: pip install scikit-learn")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    warnings.warn("nltk not installed. Basic tokenization will be used. Install with: pip install nltk")


class SentenceSummarizer:
    """
    Extractive summarizer that assigns weights to sentences and selects top-k.
    """
    
    def __init__(
        self,
        cnn_prob_weight: float = 0.25,
        textrank_weight: float = 0.35,
        tfidf_weight: float = 0.30,
        position_weight: float = 0.10,
        use_embeddings: bool = False,
        embedding_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize the summarizer with component weights.
        
        Args:
            cnn_prob_weight: Weight for CNN boundary probability (0-1)
            textrank_weight: Weight for TextRank score (0-1)
            tfidf_weight: Weight for TF-IDF similarity (0-1)
            position_weight: Weight for position-based score (0-1)
            use_embeddings: If True, use sentence-transformers for better similarity
            embedding_model_name: Model name for sentence-transformers
        """
        self.cnn_prob_weight = cnn_prob_weight
        self.textrank_weight = textrank_weight
        self.tfidf_weight = tfidf_weight
        self.position_weight = position_weight
        
        # Normalize weights to sum to 1
        total = cnn_prob_weight + textrank_weight + tfidf_weight + position_weight
        if abs(total - 1.0) > 1e-6:
            warnings.warn(f"Weights sum to {total}, normalizing to 1.0")
            self.cnn_prob_weight /= total
            self.textrank_weight /= total
            self.tfidf_weight /= total
            self.position_weight /= total
        
        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                warnings.warn(f"Failed to load sentence-transformers model: {e}. Falling back to TF-IDF.")
                self.use_embeddings = False
                self.embedding_model = None
        else:
            self.embedding_model = None
    
    def _tokenize_sentence(self, sentence: str) -> List[str]:
        """Tokenize a sentence into words."""
        if HAS_NLTK:
            try:
                tokens = word_tokenize(sentence.lower())
                # Remove stopwords and punctuation
                stop_words = set(stopwords.words('english'))
                tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
                return tokens
            except Exception:
                pass
        
        # Fallback: simple tokenization
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        return tokens
    
    def _compute_textrank_scores(self, sentences: List[str]) -> np.ndarray:
        """
        Compute TextRank scores for sentences using graph centrality.
        
        Returns:
            Array of TextRank scores (normalized to sum to 1)
        """
        if not HAS_NETWORKX:
            # Fallback: return uniform scores
            return np.ones(len(sentences)) / len(sentences)
        
        if len(sentences) < 2:
            return np.ones(len(sentences)) / len(sentences) if len(sentences) > 0 else np.array([])
        
        # Build similarity graph
        G = nx.Graph()
        G.add_nodes_from(range(len(sentences)))
        
        # Tokenize sentences
        sentence_tokens = [self._tokenize_sentence(s) for s in sentences]
        
        # Compute similarity between sentence pairs
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                tokens_i = set(sentence_tokens[i])
                tokens_j = set(sentence_tokens[j])
                
                if len(tokens_i) == 0 or len(tokens_j) == 0:
                    continue
                
                # Jaccard similarity
                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                similarity = intersection / union if union > 0 else 0.0
                
                # Add edge if similarity > threshold
                if similarity > 0.1:
                    G.add_edge(i, j, weight=similarity)
        
        # Compute PageRank (TextRank)
        try:
            pagerank_scores = nx.pagerank(G, max_iter=100)
            scores = np.array([pagerank_scores.get(i, 0.0) for i in range(len(sentences))])
        except Exception:
            # Fallback: uniform scores
            scores = np.ones(len(sentences)) / len(sentences)
        
        # Normalize
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            scores = np.ones(len(sentences)) / len(sentences)
        
        return scores
    
    def _compute_tfidf_scores(self, sentences: List[str]) -> np.ndarray:
        """
        Compute TF-IDF cosine similarity scores between sentences and document centroid.
        
        Returns:
            Array of TF-IDF scores (normalized to sum to 1)
        """
        if not HAS_SKLEARN:
            # Fallback: return uniform scores
            return np.ones(len(sentences)) / len(sentences)
        
        if len(sentences) == 0:
            return np.array([])
        
        try:
            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Compute document centroid
            centroid = np.asarray(tfidf_matrix.mean(axis=0))
            
            # Compute cosine similarity between each sentence and centroid
            similarities = cosine_similarity(tfidf_matrix, centroid).flatten()
            
            # Normalize to sum to 1
            if similarities.sum() > 0:
                similarities = similarities / similarities.sum()
            else:
                similarities = np.ones(len(sentences)) / len(sentences)
            
            return similarities
        except Exception as e:
            warnings.warn(f"TF-IDF computation failed: {e}. Using uniform scores.")
            return np.ones(len(sentences)) / len(sentences)
    
    def _compute_embedding_scores(self, sentences: List[str]) -> np.ndarray:
        """
        Compute sentence embedding similarity scores using sentence-transformers.
        
        Returns:
            Array of embedding similarity scores (normalized to sum to 1)
        """
        if not self.use_embeddings or self.embedding_model is None:
            return self._compute_tfidf_scores(sentences)  # Fallback to TF-IDF
        
        try:
            # Compute sentence embeddings
            embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
            
            # Compute document centroid
            centroid = embeddings.mean(axis=0, keepdims=True)
            
            # Compute cosine similarity
            similarities = cosine_similarity(embeddings, centroid).flatten()
            
            # Normalize to sum to 1
            if similarities.sum() > 0:
                similarities = similarities / similarities.sum()
            else:
                similarities = np.ones(len(sentences)) / len(sentences)
            
            return similarities
        except Exception as e:
            warnings.warn(f"Embedding computation failed: {e}. Falling back to TF-IDF.")
            return self._compute_tfidf_scores(sentences)
    
    def _compute_position_scores(self, sentences: List[str]) -> np.ndarray:
        """
        Compute position-based scores (leading sentences get higher scores).
        
        Returns:
            Array of position scores (normalized to sum to 1)
        """
        if len(sentences) == 0:
            return np.array([])
        
        n = len(sentences)
        # Exponential decay: first sentence gets highest score
        position_scores = np.exp(-np.arange(n) * 0.1)
        
        # Normalize to sum to 1
        if position_scores.sum() > 0:
            position_scores = position_scores / position_scores.sum()
        else:
            position_scores = np.ones(n) / n
        
        return position_scores
    
    def _compute_cnn_prob_scores(
        self,
        sentences: List[str],
        original_text: str,
        cnn_probs: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute CNN probability scores for sentences.
        
        Args:
            sentences: List of segmented sentences
            original_text: Original full text
            cnn_probs: Optional list of CNN probabilities for sentence boundaries
        
        Returns:
            Array of CNN probability scores (normalized to sum to 1)
        """
        if cnn_probs is None or len(cnn_probs) == 0:
            # No CNN probabilities available, return uniform scores
            return np.ones(len(sentences)) / len(sentences) if len(sentences) > 0 else np.array([])
        
        # Map CNN probabilities to sentences
        # If we have probabilities for boundaries, assign to the sentence that ends at that boundary
        scores = np.ones(len(sentences))
        
        # If cnn_probs has one value per sentence, use directly
        if len(cnn_probs) == len(sentences):
            scores = np.array(cnn_probs)
        elif len(cnn_probs) > len(sentences):
            # More probabilities than sentences - average or take max
            # Simple approach: take the first len(sentences) probabilities
            scores = np.array(cnn_probs[:len(sentences)])
        else:
            # Fewer probabilities - pad with 0.5 (neutral)
            scores = np.array(cnn_probs + [0.5] * (len(sentences) - len(cnn_probs)))
        
        # Normalize to sum to 1
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            scores = np.ones(len(sentences)) / len(sentences)
        
        return scores
    
    def compute_sentence_weights(
        self,
        sentences: List[str],
        original_text: str = "",
        cnn_probs: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute combined sentence weights using all available components.
        
        Args:
            sentences: List of segmented sentences
            original_text: Original full text (for context)
            cnn_probs: Optional list of CNN probabilities for sentence boundaries
        
        Returns:
            Tuple of (combined_weights, component_scores_dict)
            - combined_weights: Normalized weights summing to 1
            - component_scores_dict: Dictionary with individual component scores
        """
        if len(sentences) == 0:
            return np.array([]), {}
        
        component_scores = {}
        
        # Compute CNN probability scores
        if self.cnn_prob_weight > 0:
            cnn_scores = self._compute_cnn_prob_scores(sentences, original_text, cnn_probs)
            component_scores['cnn_prob'] = cnn_scores
        else:
            cnn_scores = np.zeros(len(sentences))
        
        # Compute TextRank scores
        if self.textrank_weight > 0:
            textrank_scores = self._compute_textrank_scores(sentences)
            component_scores['textrank'] = textrank_scores
        else:
            textrank_scores = np.zeros(len(sentences))
        
        # Compute TF-IDF or embedding scores
        if self.tfidf_weight > 0:
            if self.use_embeddings:
                similarity_scores = self._compute_embedding_scores(sentences)
                component_scores['embeddings'] = similarity_scores
            else:
                similarity_scores = self._compute_tfidf_scores(sentences)
                component_scores['tfidf'] = similarity_scores
        else:
            similarity_scores = np.zeros(len(sentences))
        
        # Compute position scores
        if self.position_weight > 0:
            position_scores = self._compute_position_scores(sentences)
            component_scores['position'] = position_scores
        else:
            position_scores = np.zeros(len(sentences))
        
        # Combine scores with weights
        combined_scores = (
            self.cnn_prob_weight * cnn_scores +
            self.textrank_weight * textrank_scores +
            self.tfidf_weight * similarity_scores +
            self.position_weight * position_scores
        )
        
        # Normalize to sum to 1
        if combined_scores.sum() > 0:
            combined_scores = combined_scores / combined_scores.sum()
        else:
            combined_scores = np.ones(len(sentences)) / len(sentences)
        
        return combined_scores, component_scores
    
    def summarize(
        self,
        sentences: List[str],
        original_text: str = "",
        cnn_probs: Optional[List[float]] = None,
        compression: Optional[float] = None,
        top_k: Optional[int] = None,
        preserve_order: bool = True
    ) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate extractive summary by selecting top sentences.
        
        Args:
            sentences: List of segmented sentences
            original_text: Original full text
            cnn_probs: Optional list of CNN probabilities
            compression: Compression ratio (0-1), e.g., 0.2 means 20% of sentences
            top_k: Number of top sentences to select (alternative to compression)
            preserve_order: If True, maintain original sentence order in summary
        
        Returns:
            Tuple of (selected_sentences, weights, component_scores_dict)
        """
        if len(sentences) == 0:
            return [], np.array([]), {}
        
        # Compute weights
        weights, component_scores = self.compute_sentence_weights(
            sentences, original_text, cnn_probs
        )
        
        # Determine number of sentences to select
        if top_k is not None:
            n_select = min(top_k, len(sentences))
        elif compression is not None:
            n_select = max(1, int(len(sentences) * compression))
        else:
            # Default: select top 30% or at least 2 sentences
            n_select = max(2, int(len(sentences) * 0.3))
        
        # Select top sentences
        if preserve_order:
            # Sort by weight but maintain original order in output
            sentence_indices = list(range(len(sentences)))
            sentence_indices.sort(key=lambda i: weights[i], reverse=True)
            selected_indices = sorted(sentence_indices[:n_select])  # Sort to preserve order
        else:
            # Select top-k by weight
            selected_indices = np.argsort(weights)[::-1][:n_select]
            selected_indices = sorted(selected_indices)  # Sort to preserve order
        
        selected_sentences = [sentences[i] for i in selected_indices]
        selected_weights = weights[selected_indices]
        
        return selected_sentences, weights, component_scores


