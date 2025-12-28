"""
Quick example demonstrating the hybrid summarization pipeline.

This script shows how to:
1. Load a document
2. Apply extractive filtering
3. Use a trained model for abstractive summarization
"""

import os
import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils.preprocessing import apply_extractive_filtering, segment_sentences


# Example legal document
EXAMPLE_DOCUMENT = """
Appeal No. LXVI of 1949. Appeal from the High Court of Judicature at Allahabad 
(Agarwala and Bind Basni Prasad JJ.) in Civil Miscellaneous Case No. 2218 of 1947. 
The facts are set out in the judgment. M.C. Setalvad, Attorney-General of India 
(G.N. Joshi, with him) for the appellant. Som Nath Sinha for respondent No. 1. 
The charge created in respect of municipal property under the U.P. Municipalities 
Act is a charge in favour of the municipality and not of the Government and the 
words "revenue payable to the Government" and "public revenue demand" in s. 3 of 
the Revenue Recovery Act must be construed as not including a municipal tax. 
The object and scheme of the Revenue Recovery Act which is an Act for the recovery 
of revenue payable to the Government clearly indicates that the Act is confined 
to the recovery of only such revenues. The definition of "public demand" in s. 3 
cannot be pressed so as to cover municipal claims when other sections in the Act 
on a harmonious construction clearly show that the legislative intention was to 
limit the scope and operation of the Act to Government revenue and Government revenue alone.
"""


def demo_extractive_filtering():
    """Demonstrate extractive filtering component."""
    print("="*80)
    print("DEMO: EXTRACTIVE FILTERING")
    print("="*80)
    
    print("\nOriginal Document:")
    print("-" * 80)
    print(EXAMPLE_DOCUMENT.strip())
    
    # Segment sentences
    sentences = segment_sentences(EXAMPLE_DOCUMENT)
    print(f"\n\nNumber of sentences: {len(sentences)}")
    
    # Apply extractive filtering
    print("\nApplying extractive summarization (top 3 sentences)...")
    extracted_text, selected_sentences, weights = apply_extractive_filtering(
        text=EXAMPLE_DOCUMENT,
        top_k=3
    )
    
    print("\nExtracted Sentences:")
    print("-" * 80)
    for i, (sent, weight) in enumerate(zip(selected_sentences, weights), 1):
        print(f"{i}. [Weight: {weight:.4f}] {sent}")
    
    print("\n\nConcatenated Extractive Summary:")
    print("-" * 80)
    print(extracted_text)
    
    return extracted_text


def demo_full_pipeline():
    """Demonstrate full pipeline (requires trained model)."""
    print("\n\n")
    print("="*80)
    print("DEMO: FULL PIPELINE (Extractive + Abstractive)")
    print("="*80)
    
    # Check if model exists
    import config
    from pathlib import Path
    
    model_dir = Path(config.MODEL_DIR)
    checkpoints = list(model_dir.glob("checkpoint_*.pt"))
    
    if not checkpoints:
        print("\n❌ No trained model found.")
        print("Please train the model first using: python train.py")
        print("\nAfter training, run this script again to see the full pipeline in action.")
        return
    
    # Use the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n✓ Found trained model: {latest_checkpoint}")
    
    # Import inference module
    from inference import SummaryGenerator
    
    # Initialize generator
    print("\nInitializing summary generator...")
    generator = SummaryGenerator(
        model_checkpoint_path=str(latest_checkpoint),
        tokenizer_path=config.TOKENIZER_MODEL_FILE
    )
    
    # Generate summary
    print("\nGenerating abstractive summary...")
    print("-" * 80)
    
    summary = generator.generate_summary(
        document=EXAMPLE_DOCUMENT,
        apply_extractive=True,
        extractive_top_k=5
    )
    
    print("\nGENERATED SUMMARY:")
    print("="*80)
    print(summary)
    print("="*80)


def demo_sentence_segmentation():
    """Demonstrate sentence segmentation."""
    print("\n\n")
    print("="*80)
    print("DEMO: SENTENCE SEGMENTATION")
    print("="*80)
    
    sentences = segment_sentences(EXAMPLE_DOCUMENT)
    
    print(f"\nDocument segmented into {len(sentences)} sentences:")
    print("-" * 80)
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYBRID EXTRACTIVE-ABSTRACTIVE SUMMARIZATION DEMO")
    print("="*80)
    
    # Demo 1: Sentence Segmentation
    demo_sentence_segmentation()
    
    # Demo 2: Extractive Filtering
    demo_extractive_filtering()
    
    # Demo 3: Full Pipeline (if model is trained)
    demo_full_pipeline()
    
    print("\n" + "="*80)
    print("DEMO COMPLETED")
    print("="*80)
    print("\nTo train the full model, run: python train.py")
    print("To generate summaries, run: python inference.py --help")
