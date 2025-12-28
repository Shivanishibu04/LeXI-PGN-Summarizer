"""
Evaluation script for Pointer-Generator Network summarization.

This script computes standard summarization metrics:
- ROUGE (1, 2, L) - Recall-Oriented Understudy for Gisting Evaluation
- BLEU - Bilingual Evaluation Understudy Score
- METEOR - Metric for Evaluation of Translation with Explicit ORdering
- BERTScore - Contextual embedding similarity
- Custom metrics: Length, Compression ratio, Abstractiveness

Usage:
    python evaluate.py --checkpoint [model_path] --input test.csv --output results.csv
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from inference import SummaryGenerator


# Try to import evaluation libraries
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("⚠️  rouge-score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    HAS_NLTK_METRICS = True
except ImportError:
    HAS_NLTK_METRICS = False
    print("⚠️  NLTK metrics not available. Install with: pip install nltk")

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("⚠️  BERTScore not installed. Install with: pip install bert-score")


class SummarizationEvaluator:
    """
    Comprehensive evaluator for abstractive summarization.
    """
    
    def __init__(self):
        """Initialize evaluator with available metrics."""
        self.has_rouge = HAS_ROUGE
        self.has_nltk = HAS_NLTK_METRICS
        self.has_bertscore = HAS_BERTSCORE
        
        if self.has_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        
        if self.has_nltk:
            self.smoothing = SmoothingFunction()
    
    def compute_rouge(self, generated: str, reference: str) -> dict:
        """
        Compute ROUGE scores (overlap-based metrics).
        
        ROUGE-1: Unigram overlap
        ROUGE-2: Bigram overlap
        ROUGE-L: Longest common subsequence
        
        Returns dict with precision, recall, f1 for each metric.
        """
        if not self.has_rouge:
            return {}
        
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }
    
    def compute_bleu(self, generated: str, reference: str) -> float:
        """
        Compute BLEU score (precision-based n-gram metric).
        
        Commonly used in machine translation, adapted for summarization.
        Returns weighted average of 1-4 gram precision.
        """
        if not self.has_nltk:
            return 0.0
        
        # Tokenize
        gen_tokens = generated.split()
        ref_tokens = reference.split()
        
        # BLEU expects reference as list of lists
        bleu = sentence_bleu(
            [ref_tokens],
            gen_tokens,
            smoothing_function=self.smoothing.method1
        )
        
        return bleu
    
    def compute_meteor(self, generated: str, reference: str) -> float:
        """
        Compute METEOR score (considers synonyms and paraphrases).
        
        More sophisticated than BLEU, accounts for:
        - Exact matches
        - Stem matches
        - Synonym matches (via WordNet)
        """
        if not self.has_nltk:
            return 0.0
        
        try:
            # Tokenize
            gen_tokens = generated.split()
            ref_tokens = reference.split()
            
            meteor = meteor_score([ref_tokens], gen_tokens)
            return meteor
        except:
            return 0.0
    
    def compute_bertscore(self, generated: str, reference: str) -> dict:
        """
        Compute BERTScore (contextual embedding similarity).
        
        Uses BERT embeddings to measure semantic similarity.
        Returns precision, recall, F1.
        """
        if not self.has_bertscore:
            return {}
        
        try:
            P, R, F1 = bert_score([generated], [reference], lang='en', verbose=False)
            return {
                'bertscore_precision': P.item(),
                'bertscore_recall': R.item(),
                'bertscore_f1': F1.item()
            }
        except:
            return {}
    
    def compute_length_metrics(self, generated: str, reference: str, source: str) -> dict:
        """
        Compute length-based metrics.
        
        - Summary length (words)
        - Compression ratio
        - Length ratio (generated/reference)
        """
        gen_len = len(generated.split())
        ref_len = len(reference.split())
        src_len = len(source.split())
        
        compression = gen_len / src_len if src_len > 0 else 0
        length_ratio = gen_len / ref_len if ref_len > 0 else 0
        
        return {
            'generated_length': gen_len,
            'reference_length': ref_len,
            'source_length': src_len,
            'compression_ratio': compression,
            'length_ratio': length_ratio
        }
    
    def compute_abstractiveness(self, generated: str, source: str) -> dict:
        """
        Compute abstractiveness metrics.
        
        - Novel n-grams: How many n-grams in summary don't appear in source
        - Measures how abstractive (vs extractive) the summary is
        """
        gen_words = set(generated.lower().split())
        src_words = set(source.lower().split())
        
        # Novel unigrams
        novel_unigrams = len(gen_words - src_words) / len(gen_words) if gen_words else 0
        
        # Novel bigrams
        gen_bigrams = set(zip(generated.lower().split()[:-1], generated.lower().split()[1:]))
        src_bigrams = set(zip(source.lower().split()[:-1], source.lower().split()[1:]))
        novel_bigrams = len(gen_bigrams - src_bigrams) / len(gen_bigrams) if gen_bigrams else 0
        
        return {
            'novel_unigrams': novel_unigrams,
            'novel_bigrams': novel_bigrams,
            'abstractiveness_score': (novel_unigrams + novel_bigrams) / 2
        }
    
    def evaluate_single(self, generated: str, reference: str, source: str) -> dict:
        """
        Evaluate a single generated summary against reference.
        
        Returns dictionary with all metrics.
        """
        metrics = {}
        
        # ROUGE scores
        metrics.update(self.compute_rouge(generated, reference))
        
        # BLEU score
        metrics['bleu'] = self.compute_bleu(generated, reference)
        
        # METEOR score
        metrics['meteor'] = self.compute_meteor(generated, reference)
        
        # BERTScore
        metrics.update(self.compute_bertscore(generated, reference))
        
        # Length metrics
        metrics.update(self.compute_length_metrics(generated, reference, source))
        
        # Abstractiveness
        metrics.update(self.compute_abstractiveness(generated, source))
        
        return metrics
    
    def aggregate_metrics(self, all_metrics: list) -> dict:
        """
        Aggregate metrics across all examples.
        
        Computes mean and std for each metric.
        """
        if not all_metrics:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        metric_names = all_metrics[0].keys()
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics if metric in m and m[metric] is not None]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated


def evaluate_model(
    checkpoint_path: str,
    input_csv: str,
    output_csv: str = None,
    max_examples: int = None
):
    """
    Evaluate model on test set.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        input_csv: Path to test CSV (must have Text and Summary columns)
        output_csv: Path to save results (optional)
        max_examples: Limit number of examples to evaluate (for testing)
    """
    print("\n" + "="*80)
    print("SUMMARIZATION EVALUATION")
    print("="*80)
    
    # Initialize generator
    print(f"\nLoading model from {checkpoint_path}...")
    generator = SummaryGenerator(
        model_checkpoint_path=checkpoint_path,
        tokenizer_path=config.TOKENIZER_MODEL_FILE
    )
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = SummarizationEvaluator()
    
    print(f"\nAvailable metrics:")
    print(f"  ROUGE: {'✓' if evaluator.has_rouge else '✗'}")
    print(f"  BLEU/METEOR: {'✓' if evaluator.has_nltk else '✗'}")
    print(f"  BERTScore: {'✓' if evaluator.has_bertscore else '✗'}")
    print(f"  Length metrics: ✓")
    print(f"  Abstractiveness: ✓")
    
    # Load test data
    print(f"\nLoading test data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if max_examples:
        df = df.head(max_examples)
    
    print(f"Evaluating on {len(df)} examples...")
    
    # Evaluate
    all_metrics = []
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        source_text = str(row['Text'])
        reference_summary = str(row['Summary'])
        
        # Generate summary
        generated_summary = generator.generate_summary(
            document=source_text,
            apply_extractive=True
        )
        
        # Compute metrics
        metrics = evaluator.evaluate_single(
            generated=generated_summary,
            reference=reference_summary,
            source=source_text
        )
        
        all_metrics.append(metrics)
        
        # Store result
        results.append({
            'source': source_text,
            'reference': reference_summary,
            'generated': generated_summary,
            **metrics
        })
    
    # Aggregate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    aggregated = evaluator.aggregate_metrics(all_metrics)
    
    # Print results
    print("\n📊 ROUGE Scores (overlap-based):")
    if 'rouge1_f1_mean' in aggregated:
        print(f"  ROUGE-1: {aggregated['rouge1_f1_mean']:.4f} (±{aggregated.get('rouge1_f1_std', 0):.4f})")
        print(f"  ROUGE-2: {aggregated['rouge2_f1_mean']:.4f} (±{aggregated.get('rouge2_f1_std', 0):.4f})")
        print(f"  ROUGE-L: {aggregated['rougeL_f1_mean']:.4f} (±{aggregated.get('rougeL_f1_std', 0):.4f})")
    
    print("\n📊 Other Metrics:")
    if 'bleu_mean' in aggregated:
        print(f"  BLEU: {aggregated['bleu_mean']:.4f} (±{aggregated.get('bleu_std', 0):.4f})")
    if 'meteor_mean' in aggregated:
        print(f"  METEOR: {aggregated['meteor_mean']:.4f} (±{aggregated.get('meteor_std', 0):.4f})")
    if 'bertscore_f1_mean' in aggregated:
        print(f"  BERTScore: {aggregated['bertscore_f1_mean']:.4f} (±{aggregated.get('bertscore_f1_std', 0):.4f})")
    
    print("\n📊 Length Metrics:")
    if 'generated_length_mean' in aggregated:
        print(f"  Avg generated length: {aggregated['generated_length_mean']:.1f} words")
        print(f"  Avg reference length: {aggregated['reference_length_mean']:.1f} words")
        print(f"  Compression ratio: {aggregated['compression_ratio_mean']:.3f}")
        print(f"  Length ratio (gen/ref): {aggregated['length_ratio_mean']:.3f}")
    
    print("\n📊 Abstractiveness:")
    if 'abstractiveness_score_mean' in aggregated:
        print(f"  Novel unigrams: {aggregated['novel_unigrams_mean']:.3f}")
        print(f"  Novel bigrams: {aggregated['novel_bigrams_mean']:.3f}")
        print(f"  Abstractiveness: {aggregated['abstractiveness_score_mean']:.3f}")
    
    # Save results if output path provided
    if output_csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n✓ Detailed results saved to: {output_csv}")
        
        # Also save aggregated metrics
        metrics_file = Path(output_csv).parent / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"✓ Aggregated metrics saved to: {metrics_file}")
    
    print("\n" + "="*80)
    
    return aggregated, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained summarization model")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to test CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results'
    )
    
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum number of examples to evaluate (for testing)'
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        args.output = str(config.RESULTS_DIR / "evaluation_results.csv")
    
    # Run evaluation
    evaluate_model(
        checkpoint_path=args.checkpoint,
        input_csv=args.input,
        output_csv=args.output,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()
