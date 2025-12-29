"""
Inference script for generating summaries using trained Pointer-Generator Network.

This script:
1. Loads a trained model checkpoint
2. Applies extractive filtering to input documents
3. Generates abstractive summaries using the PGN
4. Evaluates on test set (optional)
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from models import PointerGeneratorNetwork
from data_utils import load_sentencepiece_tokenizer
from data_utils.preprocessing import apply_extractive_filtering, encode_with_oov, decode_with_oov


class SummaryGenerator:
    """
    Wrapper class for generating summaries with trained PGN.
    """
    
    def __init__(
        self,
        model_checkpoint_path: str,
        tokenizer_path: str,
        device: str = None
    ):
        """
        Initialize generator.
        
        Args:
            model_checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to SentencePiece model
            device: Device to use (cuda/cpu)
        """
        self.device = device if device else str(config.DEVICE)
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = load_sentencepiece_tokenizer(tokenizer_path)
        
        # Initialize model
        print(f"Initializing model...")
        self.model = PointerGeneratorNetwork(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            encoder_hidden_dim=config.ENCODER_HIDDEN_DIM,
            decoder_hidden_dim=config.DECODER_HIDDEN_DIM,
            attention_dim=config.ATTENTION_DIM,
            encoder_num_layers=config.ENCODER_NUM_LAYERS,
            decoder_num_layers=config.DECODER_NUM_LAYERS,
            dropout=config.ENCODER_DROPOUT,
            pad_idx=config.PAD_IDX,
            use_coverage=config.USE_COVERAGE
        )
        
        # Load checkpoint
        print(f"Loading model checkpoint from {model_checkpoint_path}...")
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    
    def generate_summary(
        self,
        document: str,
        max_length: int = None,
        apply_extractive: bool = True,
        extractive_top_k: int = None
    ) -> str:
        """
        Generate summary for a single document.
        
        Args:
            document: Input document text
            max_length: Maximum summary length
            apply_extractive: Whether to apply extractive filtering first
            extractive_top_k: Number of sentences to extract
            
        Returns:
            Generated summary text
        """
        if max_length is None:
            max_length = config.MAX_DECODE_LEN
        
        if extractive_top_k is None:
            extractive_top_k = config.EXTRACTIVE_TOP_K
        
        # Step 1: Apply extractive filtering if requested
        if apply_extractive:
            extracted_text, _, _ = apply_extractive_filtering(
                text=document,
                top_k=extractive_top_k
            )
            
            if not extracted_text:
                extracted_text = document
        else:
            extracted_text = document
        
        # Step 2: Encode source text
        encoder_ids, encoder_extended_ids, oov_words = encode_with_oov(
            text=extracted_text,
            tokenizer=self.tokenizer,
            vocab_size=config.VOCAB_SIZE,
            max_len=config.MAX_ENCODER_LEN
        )
        
        # Convert to tensors
        encoder_input = torch.tensor([encoder_ids], dtype=torch.long).to(self.device)
        encoder_input_extended = torch.tensor([encoder_extended_ids], dtype=torch.long).to(self.device)
        encoder_lengths = torch.tensor([len(encoder_ids)], dtype=torch.long).to(self.device)
        
        # Create extra_zeros for OOV
        if len(oov_words) > 0:
            extra_zeros = torch.zeros(1, len(oov_words)).to(self.device)
        else:
            extra_zeros = None
        
        # Step 3: Generate summary
        with torch.no_grad():
            generated_ids = self.model.generate(
                encoder_input=encoder_input,
                encoder_lengths=encoder_lengths,
                encoder_input_extended=encoder_input_extended,
                extra_zeros=extra_zeros,
                max_len=max_length,
                bos_idx=config.BOS_IDX,
                eos_idx=config.EOS_IDX
            )
        
        # Step 4: Decode to text
        generated_ids = generated_ids[0].cpu().tolist()
        
        # Remove BOS, EOS, and PAD tokens
        generated_ids = [
            id for id in generated_ids
            if id not in [config.PAD_IDX, config.BOS_IDX, config.EOS_IDX]
        ]
        
        # Decode
        summary = decode_with_oov(
            token_ids=generated_ids,
            tokenizer=self.tokenizer,
            oov_words=oov_words,
            vocab_size=config.VOCAB_SIZE
        )
        
        return summary.strip()
    
    def generate_for_dataset(
        self,
        csv_path: str,
        output_path: str,
        max_length: int = None
    ):
        """
        Generate summaries for entire dataset and save to CSV.
        
        Args:
            csv_path: Path to input CSV with 'Text' column
            output_path: Path to save output CSV
            max_length: Maximum summary length
        """
        print(f"\nGenerating summaries for dataset: {csv_path}")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        if 'Text' not in df.columns:
            raise ValueError("CSV must contain 'Text' column")
        
        # Generate summaries
        summaries = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
            document = str(row['Text'])
            summary = self.generate_summary(
                document=document,
                max_length=max_length
            )
            summaries.append(summary)
        
        # Add to dataframe
        df['Generated_Summary'] = summaries
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate summaries using trained Pointer-Generator Network"
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file (for batch processing)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Direct text input (for single document)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=config.MAX_DECODE_LEN,
        help='Maximum summary length'
    )
    
    parser.add_argument(
        '--no-extractive',
        action='store_true',
        help='Skip extractive filtering step'
    )
    
    args = parser.parse_args()
    
    # Smart config switching: Check if using fast model
    if 'fast' in args.checkpoint.lower() or (args.output and 'fast' in args.output.lower()):
        print("\n" + "!"*80)
        print("DETECTED FAST MODEL/OUTPUT PATH - SWITCHING TO CONFIG_FAST")
        print("!"*80 + "\n")
        import config_fast
        # Update attributes of the imported config module
        for attribute in dir(config_fast):
            if not attribute.startswith('__'):
                setattr(config, attribute, getattr(config_fast, attribute))
    
    # Initialize generator
    tokenizer_path = config.TOKENIZER_MODEL_FILE
    
    generator = SummaryGenerator(
        model_checkpoint_path=args.checkpoint,
        tokenizer_path=tokenizer_path
    )
    
    # Single document mode
    if args.text:
        print("\n" + "="*80)
        print("GENERATING SUMMARY")
        print("="*80)
        print(f"\nInput document:\n{args.text[:500]}...")
        
        summary = generator.generate_summary(
            document=args.text,
            max_length=args.max_length,
            apply_extractive=not args.no_extractive
        )
        
        print(f"\nGenerated summary:\n{summary}")
        print("\n" + "="*80)
    
    # Batch mode
    elif args.input:
        if not args.output:
            args.output = str(config.RESULTS_DIR / "generated_summaries.csv")
        
        generator.generate_for_dataset(
            csv_path=args.input,
            output_path=args.output,
            max_length=args.max_length
        )
    
    else:
        print("Error: Must provide either --text or --input argument")
        parser.print_help()


if __name__ == "__main__":
    main()
