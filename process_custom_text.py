import os
import sys
import torch
import warnings

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
# Using the same "smart config switching" logic as inference.py
from inference import SummaryGenerator
from data_utils.preprocessing import segment_sentences, apply_extractive_filtering

# The input text provided by the user
INPUT_TEXT = """The present appeal arises out of a land acquisition proceeding initiated by the State 
Government in the year 2012 for the purpose of constructing a public highway. 
The Land Acquisition Officer had initially awarded a compensation of Rs. 5 Lakhs per acre. 
The landowners, being dissatisfied with this amount, sought a reference to the District 
Judge under Section 18 of the Land Acquisition Act. The District Judge enhanced the 
compensation to Rs. 15 Lakhs per acre, citing the proximity of the land to the developing 
industrial hub. The State Government challenged this enhancement in the High Court, 
contending that the market value was inflated and based on speculative future developments. 
The High Court, after reviewing the sale deeds of comparable lands in the vicinity, 
partially allowed the appeal and reduced the compensation to Rs. 10 Lakhs per acre. 
The landowners have now approached this Court, arguing that the High Court failed to 
consider the potential non-agricultural use of the land. After hearing both parties, 
we find that the High Court's valuation was based on a sound appreciation of the evidence 
on record. The potential for future development was already factored into the market value 
by the reference court. Consequently, we see no reason to interfere with the High Court’s 
judgment. The appeals are accordingly dismissed, and no order as to costs is passed."""

def main():
    # Paths for the fast model
    checkpoint_path = "pgn_output_fast/models/checkpoint_epoch10_step600.pt"
    tokenizer_path = "pgn_output_fast/tokenizer/legal_bpe_fast.model"
    
    print("Initializing Summary Generator...")
    # SummaryGenerator handles config switching internally based on path
    generator = SummaryGenerator(
        model_checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path
    )
    
    # 1. Get segmented sentences
    print("Segmenting sentences...")
    sentences = segment_sentences(INPUT_TEXT)
    
    # 2. Get extractive summary
    print("Generating extractive summary...")
    # top_k=5 to get a good subset
    extracted_text, selected_sentences, _ = apply_extractive_filtering(INPUT_TEXT, top_k=5)
    
    # 3. Get final abstractive summary
    print("Generating final abstractive summary...")
    # apply_extractive=True tells the generator to use the extractive step first
    abstractive_summary = generator.generate_summary(INPUT_TEXT, apply_extractive=True)
    
    # 4. Prepare output
    output = []
    output.append("="*80)
    output.append("DETAILED SUMMARIZATION OUTPUT")
    output.append("="*80)
    
    output.append("\n[1] ALL SEGMENTED SENTENCES:")
    for i, s in enumerate(sentences, 1):
        output.append(f"  {i}. {s}")
        
    output.append("\n[2] EXTRXATIVE SUMMARY GENERATED (Top Sentences):")
    output.append(f"  {extracted_text}")
    
    output.append("\n[3] FINAL ABSTARCTIVE SUMMARY:")
    output.append(f"  {abstractive_summary}")
    
    final_text = "\n".join(output)
    
    # Write to file requested by user (keeping typo 'outpuut_summary')
    filename = "outpuut_summary.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"\nProcessing complete. Results saved to {filename}")
    print("\n" + "="*80)
    print("PREVIEW:")
    print("="*80)
    print(final_text[:1000] + "...")

if __name__ == "__main__":
    main()
