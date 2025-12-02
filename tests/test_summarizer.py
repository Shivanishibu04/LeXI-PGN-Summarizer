"""
Unit tests for the summarization module.
"""

import pytest
import numpy as np
from src.summarizer import SentenceSummarizer


class TestSentenceSummarizer:
    """Test cases for SentenceSummarizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_text = (
            "The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), "
            "established the principle of judicial review. "
            "This principle is outlined in § 1.3(a) of the legal code. "
            "The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. "
            "All proceedings were documented by the F.B.I. for review."
        )
        self.sample_sentences = [
            "The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review.",
            "This principle is outlined in § 1.3(a) of the legal code.",
            "The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001.",
            "All proceedings were documented by the F.B.I. for review."
        ]
        self.summarizer = SentenceSummarizer()
    
    def test_weight_normalization(self):
        """Test that weights sum to 1 (within small epsilon)."""
        weights, _ = self.summarizer.compute_sentence_weights(
            self.sample_sentences, self.sample_text
        )
        
        assert len(weights) == len(self.sample_sentences)
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights sum to {weights.sum()}, expected 1.0"
        assert all(weights >= 0), "All weights should be non-negative"
        assert all(weights <= 1), "All weights should be <= 1"
    
    def test_compression_ratio(self):
        """Test that compression ratio produces correct number of sentences."""
        compression = 0.5
        selected, weights, _ = self.summarizer.summarize(
            self.sample_sentences,
            self.sample_text,
            compression=compression
        )
        
        expected_count = max(1, int(len(self.sample_sentences) * compression))
        assert len(selected) == expected_count, \
            f"Expected {expected_count} sentences, got {len(selected)}"
        assert len(selected) <= len(self.sample_sentences)
    
    def test_top_k_selection(self):
        """Test that top_k selects correct number of sentences."""
        top_k = 2
        selected, weights, _ = self.summarizer.summarize(
            self.sample_sentences,
            self.sample_text,
            top_k=top_k
        )
        
        assert len(selected) == top_k, f"Expected {top_k} sentences, got {len(selected)}"
        assert len(selected) <= len(self.sample_sentences)
    
    def test_preserve_order(self):
        """Test that preserve_order maintains original sentence order."""
        selected, weights, _ = self.summarizer.summarize(
            self.sample_sentences,
            self.sample_text,
            top_k=2,
            preserve_order=True
        )
        
        # Check that selected sentences appear in original order
        selected_indices = [self.sample_sentences.index(s) for s in selected]
        assert selected_indices == sorted(selected_indices), \
            "Sentences should be in original order when preserve_order=True"
    
    def test_empty_input(self):
        """Test handling of empty input."""
        weights, _ = self.summarizer.compute_sentence_weights([], "")
        assert len(weights) == 0
        
        selected, weights, _ = self.summarizer.summarize([], "")
        assert len(selected) == 0
    
    def test_single_sentence(self):
        """Test handling of single sentence."""
        single_sent = ["This is a single sentence."]
        weights, _ = self.summarizer.compute_sentence_weights(single_sent, "")
        
        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6, "Single sentence should have weight 1.0"
        
        selected, _, _ = self.summarizer.summarize(single_sent, "", top_k=1)
        assert len(selected) == 1
        assert selected[0] == single_sent[0]
    
    def test_cnn_prob_integration(self):
        """Test that CNN probabilities are integrated correctly."""
        cnn_probs = [0.9, 0.7, 0.8, 0.6]  # Mock CNN probabilities
        weights, component_scores = self.summarizer.compute_sentence_weights(
            self.sample_sentences,
            self.sample_text,
            cnn_probs=cnn_probs
        )
        
        assert len(weights) == len(self.sample_sentences)
        assert 'cnn_prob' in component_scores or self.summarizer.cnn_prob_weight == 0
    
    def test_component_scores_present(self):
        """Test that component scores are returned."""
        weights, component_scores = self.summarizer.compute_sentence_weights(
            self.sample_sentences,
            self.sample_text
        )
        
        # Check that at least some component scores are present
        assert len(component_scores) > 0, "Component scores should be present"
        
        # Check that component scores have correct length
        for component_name, scores in component_scores.items():
            assert len(scores) == len(self.sample_sentences), \
                f"Component {component_name} should have {len(self.sample_sentences)} scores"
    
    def test_default_compression(self):
        """Test that default compression works when neither compression nor top_k specified."""
        selected, weights, _ = self.summarizer.summarize(
            self.sample_sentences,
            self.sample_text
        )
        
        # Default should select at least 2 sentences or 30% of sentences
        expected_min = max(2, int(len(self.sample_sentences) * 0.3))
        assert len(selected) >= expected_min, \
            f"Default should select at least {expected_min} sentences"
    
    def test_custom_weights(self):
        """Test that custom weights work correctly."""
        custom_summarizer = SentenceSummarizer(
            cnn_prob_weight=0.1,
            textrank_weight=0.5,
            tfidf_weight=0.3,
            position_weight=0.1
        )
        
        weights, _ = custom_summarizer.compute_sentence_weights(
            self.sample_sentences,
            self.sample_text
        )
        
        assert len(weights) == len(self.sample_sentences)
        assert abs(weights.sum() - 1.0) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


