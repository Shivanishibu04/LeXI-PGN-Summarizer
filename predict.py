# In predict.py (in the root LexiDesk folder)

import argparse
import sys
import joblib
import re
import torch
import pandas as pd
from src.feature_extractor import token_to_features, add_neighboring_token_features
from src.cnn_model import LegalSBD_CNN # Import the CNN class definition
from src.crf_model import CONTEXT_WINDOW_SIZE, DELIMITERS # Import constants
from src.summarizer import SentenceSummarizer

# --- 1. Define Model Paths ---
BASELINE_MODEL_PATH = 'saved_models/crf_baseline_model.joblib'
HYBRID_MODEL_PATH = 'saved_models/crf_hybrid_model.joblib'
CNN_MODEL_PATH = 'saved_models/cnn_model.pth'

# --- 2. Load All Trained Models ---
def load_models():
    """Load all trained models and return them along with necessary components."""
    print("Loading all trained models...")
    try:
        # Load the CRF models
        baseline_crf_model = joblib.load(BASELINE_MODEL_PATH)
        hybrid_crf_model = joblib.load(HYBRID_MODEL_PATH)
        
        # --- Load components needed for the Hybrid Model's feature extraction ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Rebuild the character vocabulary exactly as done in training
        train_df = pd.read_csv('data/processed/train_data.csv')
        all_chars = set()
        for index, row in train_df.iterrows():
            all_chars.update(str(row['left_context']))
            all_chars.update(str(row['delimiter']))
            all_chars.update(str(row['right_context']))
        char_to_idx = {char: i+2 for i, char in enumerate(sorted(list(all_chars)))}
        char_to_idx['<PAD>'] = 0
        char_to_idx['<UNK>'] = 1
        vocab_size = len(char_to_idx)
        
        # Load the CNN model architecture and its trained weights
        cnn_model = LegalSBD_CNN(
            vocab_size=vocab_size, 
            embedding_dim=128, 
            num_filters=6, 
            kernel_size=5, 
            hidden_dim=250, 
            dropout_prob=0.2
        ).to(device)
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
        cnn_model.eval()
        
        print("All models loaded successfully.")
        return baseline_crf_model, hybrid_crf_model, cnn_model, char_to_idx, device, vocab_size

    except FileNotFoundError as e:
        print(f"Warning: A model file was not found: {e}")
        print("Some features may be unavailable. Continuing with available models...")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Warning: Error loading models: {e}")
        print("Some features may be unavailable. Continuing with available models...")
        return None, None, None, None, None, None

# Load models at module level (for backward compatibility)
baseline_crf_model, hybrid_crf_model, cnn_model, char_to_idx, device, vocab_size = load_models()

# --- 3. Prediction and Feature Extraction Functions ---

def get_cnn_prediction_from_context(text, token_start_idx, cnn_model, char_to_idx, device):
    """
    (This function remains the same)
    Identical to the function in crf_model.py. Gets the CNN's prediction.
    """
    if cnn_model is None or char_to_idx is None or device is None:
        return 0.0
    
    token = text[token_start_idx]
    if token not in DELIMITERS:
        return 0.0
    start_left = max(0, token_start_idx - CONTEXT_WINDOW_SIZE)
    left_context = text[start_left : token_start_idx]
    end_right = token_start_idx + 1 + CONTEXT_WINDOW_SIZE
    right_context = text[token_start_idx + 1 : end_right]
    sample_text = left_context + token + right_context
    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1
    pad_idx = char_to_idx['<PAD>']
    indexed_text = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in sample_text]
    padded_text = indexed_text[:max_len] + [pad_idx] * (max_len - len(indexed_text))
    text_tensor = torch.tensor([padded_text], dtype=torch.long).to(device)
    with torch.no_grad(): 
        prediction = cnn_model(text_tensor).item()
    return prediction

 # --- CHANGE HERE: REPLACING THE ENTIRE segment_text FUNCTION ---

def segment_text(text, model, use_hybrid_features=False, return_cnn_probs=False):
    """
    Takes raw text and uses a trained CRF model to split it into sentences.
    This version uses the CORRECT tokenizer and has ROBUST sentence reconstruction.
    
    Args:
        text: Input text to segment
        model: CRF model to use for segmentation
        use_hybrid_features: If True, use CNN features in hybrid model
        return_cnn_probs: If True, also return CNN probabilities for sentence boundaries
    
    Returns:
        If return_cnn_probs is False: List of sentences
        If return_cnn_probs is True: Tuple of (sentences, cnn_probs)
    """
    # 1. Get tokens and their spans using the CORRECT regex that separates punctuation.
    tokens_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)]
    
    if not tokens_with_spans:
        return ([], []) if return_cnn_probs else []

    # 2. Extract features for the tokens, exactly as done in training.
    sentence_features = []
    cnn_probs_list = []
    for token, start, end in tokens_with_spans:
        features = token_to_features(token, text, start, end)
        
        # If using the hybrid model, generate and add the CNN feature
        if use_hybrid_features and token in DELIMITERS and cnn_model is not None:
            # We need the character index, which is the start of the token span
            cnn_prob = get_cnn_prediction_from_context(text, start, cnn_model, char_to_idx, device)
            features['cnn_prob'] = round(cnn_prob, 4)
            if return_cnn_probs:
                cnn_probs_list.append(cnn_prob)
        elif return_cnn_probs:
            cnn_probs_list.append(0.0)  # No CNN prob available
        
        sentence_features.append(features)
    
    sentence_features = add_neighboring_token_features(sentence_features)

    # 3. Predict the labels ('B' for Boundary, 'O' for Other) for the sequence
    labels = model.predict([sentence_features])[0]

    # 4. Reconstruct sentences based on the predicted 'B' labels. This logic is much cleaner.
    sentences = []
    sentence_cnn_probs = []  # CNN probs for sentence boundaries
    current_sentence_start_index = 0
    for i, label in enumerate(labels):
        # A 'B' label means the token AT THIS INDEX is the end of a sentence.
        if label == 'B':
            # The sentence runs from the start index up to and including the current token.
            sentence_end_char_index = tokens_with_spans[i][2] # Get the 'end' span of the boundary token
            sentence_start_char_index = tokens_with_spans[current_sentence_start_index][1] # Get 'start' span
            
            # Slice the original text to get the sentence perfectly formatted.
            sentence = text[sentence_start_char_index:sentence_end_char_index].strip()
            sentences.append(sentence)
            
            # Store CNN prob for this boundary if available
            if return_cnn_probs and i < len(cnn_probs_list):
                sentence_cnn_probs.append(cnn_probs_list[i])
            
            # The next sentence will start at the next token.
            current_sentence_start_index = i + 1
    
    # After the loop, check if there are any leftover tokens that form a final sentence.
    if current_sentence_start_index < len(tokens_with_spans):
        # The final sentence runs from the start index to the very end of the text.
        sentence_start_char_index = tokens_with_spans[current_sentence_start_index][1]
        sentence = text[sentence_start_char_index:].strip()
        if sentence: # Make sure it's not just whitespace
             sentences.append(sentence)
             if return_cnn_probs:
                 # Use average of remaining probs or 0.5 as default
                 if len(cnn_probs_list) > len(sentence_cnn_probs):
                     sentence_cnn_probs.append(0.5)
                 elif len(sentence_cnn_probs) == 0:
                     sentence_cnn_probs.append(0.5)
    
    if return_cnn_probs:
        # Ensure we have one prob per sentence
        while len(sentence_cnn_probs) < len(sentences):
            sentence_cnn_probs.append(0.5)
        return sentences, sentence_cnn_probs
    else:
        return sentences

# --- END OF CHANGES ---


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='LeXIDesk: Legal text sentence boundary detection and summarization'
    )
    parser.add_argument(
        '--summarize',
        action='store_true',
        help='Enable summarization mode'
    )
    parser.add_argument(
        '--text-file',
        type=str,
        help='Path to input text file'
    )
    parser.add_argument(
        '--stdin',
        action='store_true',
        help='Read input from stdin'
    )
    parser.add_argument(
        '--compression',
        type=float,
        help='Compression ratio (0-1), e.g., 0.2 means 20%% of sentences'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        help='Number of top sentences to select'
    )
    parser.add_argument(
        '--preserve-order',
        action='store_true',
        default=True,
        help='Preserve original sentence order in summary (default: True)'
    )
    parser.add_argument(
        '--use-hybrid',
        action='store_true',
        default=True,
        help='Use hybrid CNN-CRF model (default: True)'
    )
    parser.add_argument(
        '--use-embeddings',
        action='store_true',
        help='Use sentence-transformers for better similarity (requires sentence-transformers)'
    )
    
    args = parser.parse_args()
    
    # Load models if not already loaded
    global baseline_crf_model, hybrid_crf_model, cnn_model, char_to_idx, device, vocab_size
    if baseline_crf_model is None:
        baseline_crf_model, hybrid_crf_model, cnn_model, char_to_idx, device, vocab_size = load_models()
    
    if hybrid_crf_model is None:
        print("Warning: Hybrid model not available. Using baseline model.")
        args.use_hybrid = False
    
    # Read input text
    if args.stdin:
        text = sys.stdin.read()
    elif args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.text_file}")
            sys.exit(1)
    else:
        # Default: use sample text
        text = """
The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review. This principle is outlined in § 1.3(a) of the legal code. The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. All proceedings were documented by the F.B.I. for review.
"""

    text = text.strip()
    if not text:
        print("Error: No input text provided.")
        sys.exit(1)
    
    # Select model
    if baseline_crf_model is None:
        print("Error: No models available. Please train models first using: python -m src.crf_model")
        sys.exit(1)
    
    model = hybrid_crf_model if args.use_hybrid and hybrid_crf_model is not None else baseline_crf_model
    use_hybrid = args.use_hybrid and hybrid_crf_model is not None and cnn_model is not None
    
    if args.summarize:
        # Summarization mode
        print("\n" + "="*50)
        print("--- Summarization Mode ---")
        print(f"\nOriginal Text:\n---\n{text}\n---")
        
        # Segment text and get CNN probabilities
        print("\nSegmenting input...")
        sentences, cnn_probs = segment_text(
            text, model, use_hybrid_features=use_hybrid, return_cnn_probs=True
        )
        
        if not sentences:
            print("Error: No sentences detected.")
            sys.exit(1)
        
        print(f"\nSegmented Sentences:")
        for i, sent in enumerate(sentences):
            print(f"[{i+1}] {sent}")
        
        # Initialize summarizer
        summarizer = SentenceSummarizer(use_embeddings=args.use_embeddings)
        
        print("\nComputing sentence weights (components: cnn_prob={:.2f}, textrank={:.2f}, tfidf={:.2f}, position={:.2f})...".format(
            summarizer.cnn_prob_weight, summarizer.textrank_weight,
            summarizer.tfidf_weight, summarizer.position_weight
        ))
        
        # Compute weights
        weights, component_scores = summarizer.compute_sentence_weights(
            sentences, text, cnn_probs if use_hybrid else None
        )
        
        print("\nWeights (normalized):")
        for i, (sent, weight) in enumerate(zip(sentences, weights)):
            print(f"[{i+1}] {weight:.3f} ({weight*100:.1f}%)")
        
        # Generate summary
        selected_sentences, all_weights, _ = summarizer.summarize(
            sentences,
            text,
            cnn_probs if use_hybrid else None,
            compression=args.compression,
            top_k=args.top_k,
            preserve_order=args.preserve_order
        )
        
        # Determine summary length info
        if args.top_k:
            summary_info = f"top {args.top_k} sentences"
        elif args.compression:
            summary_info = f"compression={args.compression}"
        else:
            summary_info = f"top {len(selected_sentences)} sentences (default)"
        
        print(f"\nSummary ({summary_info}):")
        for i, sent in enumerate(selected_sentences):
            # Find original index
            orig_idx = sentences.index(sent) + 1
            weight = weights[orig_idx - 1]
            print(f"{i+1}) [{orig_idx}] {sent} — {weight*100:.1f}%")
        
        print("\n" + "="*50)
    else:
        # Segmentation-only mode (backward compatibility)
        print("\n" + "="*50)
        print("--- Segmenting Sample Legal Text ---")
        print(f"\nOriginal Text:\n---\n{text}\n---")

        # Run prediction with the Baseline Model
        if baseline_crf_model is not None:
            print("\n--- Detected Sentences (Baseline CRF Model) ---")
            baseline_sentences = segment_text(text, baseline_crf_model, use_hybrid_features=False)
            for i, sent in enumerate(baseline_sentences):
                print(f"[{i+1}]: {sent}")

        # Run prediction with the Hybrid Model
        if use_hybrid and hybrid_crf_model is not None:
            print("\n--- Detected Sentences (Hybrid CNN-CRF Model) ---")
            hybrid_sentences = segment_text(text, hybrid_crf_model, use_hybrid_features=True)
            for i, sent in enumerate(hybrid_sentences):
                print(f"[{i+1}]: {sent}")

        print("\n" + "="*50)


if __name__ == '__main__':
    main()