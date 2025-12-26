# In predict.py (in the root LexiDesk folder)

import argparse
import sys
import joblib
import re
import torch
import pandas as pd
import warnings
from src.feature_extractor import token_to_features, add_neighboring_token_features
from src.cnn_model import LegalSBD_CNN  # Import the CNN class definition
from src.crf_model import CONTEXT_WINDOW_SIZE, DELIMITERS  # Import constants
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
    Identical to the function in crf_model.py. Gets the CNN's prediction.
    """
    if cnn_model is None or char_to_idx is None or device is None:
        return 0.0

    # token_start_idx is the character index of the delimiter
    if token_start_idx < 0 or token_start_idx >= len(text):
        return 0.0

    token = text[token_start_idx]
    if token not in DELIMITERS:
        return 0.0
    start_left = max(0, token_start_idx - CONTEXT_WINDOW_SIZE)
    left_context = text[start_left: token_start_idx]
    end_right = token_start_idx + 1 + CONTEXT_WINDOW_SIZE
    right_context = text[token_start_idx + 1: end_right]
    sample_text = left_context + token + right_context
    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1
    pad_idx = char_to_idx['<PAD>']
    indexed_text = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in sample_text]
    padded_text = indexed_text[:max_len] + [pad_idx] * (max_len - len(indexed_text))
    text_tensor = torch.tensor([padded_text], dtype=torch.long).to(device)
    with torch.no_grad():
        prediction = cnn_model(text_tensor).item()
    return prediction


# --- segment_text function (robust) ---
def segment_text(text, model, use_hybrid_features=False, return_cnn_probs=False):
    """
    Takes raw text and uses a trained CRF model to split it into sentences.

    Args:
        text: Input text to segment
        model: CRF model to use for segmentation
        use_hybrid_features: If True, use CNN features in hybrid model
        return_cnn_probs: If True, also return CNN probabilities for sentence boundaries

    Returns:
        If return_cnn_probs is False: List of sentences
        If return_cnn_probs is True: Tuple of (sentences, cnn_probs)
    """
    # Tokenize into tokens and capture spans (keeps punctuation separate)
    tokens_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)]

    if not tokens_with_spans:
        return ([], []) if return_cnn_probs else []

    # Extract features
    sentence_features = []
    cnn_probs_list = []
    for token, start, end in tokens_with_spans:
        features = token_to_features(token, text, start, end)

        # If using the hybrid model, generate and add the CNN feature
        if use_hybrid_features and token in DELIMITERS and cnn_model is not None:
            # Use the character index for the token start
            cnn_prob = get_cnn_prediction_from_context(text, start, cnn_model, char_to_idx, device)
            features['cnn_prob'] = round(cnn_prob, 4)
            if return_cnn_probs:
                cnn_probs_list.append(cnn_prob)
        elif return_cnn_probs:
            cnn_probs_list.append(0.0)  # No CNN prob available

        sentence_features.append(features)

    sentence_features = add_neighboring_token_features(sentence_features)

    # Predict labels
    try:
        labels = model.predict([sentence_features])[0]
    except Exception as e:
        # If model fails, fallback to simple rule-based splitter
        warnings.warn(f"CRF predict failed: {e}. Using fallback splitter.")
        # simple fallback: split on sentence-ending punctuation
        fallback = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if return_cnn_probs:
            return fallback, [0.5] * len(fallback)
        return fallback

    # Reconstruct sentences from labels
    sentences = []
    sentence_cnn_probs = []
    current_sentence_start_index = 0
    for i, label in enumerate(labels):
        if label == 'B':
            sentence_end_char_index = tokens_with_spans[i][2]
            sentence_start_char_index = tokens_with_spans[current_sentence_start_index][1]
            sentence = text[sentence_start_char_index:sentence_end_char_index].strip()
            sentences.append(sentence)
            if return_cnn_probs and i < len(cnn_probs_list):
                sentence_cnn_probs.append(cnn_probs_list[i])
            current_sentence_start_index = i + 1

    # Handle leftover tokens as final sentence
    if current_sentence_start_index < len(tokens_with_spans):
        sentence_start_char_index = tokens_with_spans[current_sentence_start_index][1]
        sentence = text[sentence_start_char_index:].strip()
        if sentence:
            sentences.append(sentence)
            if return_cnn_probs:
                # fallback prob for final sentence
                if len(cnn_probs_list) > len(sentence_cnn_probs):
                    sentence_cnn_probs.append(0.5)
                elif len(sentence_cnn_probs) == 0:
                    sentence_cnn_probs.append(0.5)

    if return_cnn_probs:
        while len(sentence_cnn_probs) < len(sentences):
            sentence_cnn_probs.append(0.5)
        return sentences, sentence_cnn_probs
    else:
        return sentences


# --- Helper: get multiline input from user (interactive) ---
def get_multiline_input(prompt="Enter/Paste text (end with an empty line):"):
    print(prompt)
    lines = []
    try:
        while True:
            line = input()
            # empty line ends input
            if line.strip() == "":
                break
            lines.append(line)
    except EOFError:
        # In case stdin closed
        pass
    return "\n".join(lines).strip()


def run_segmentation_and_optional_summary(text, do_summarize=False, use_hybrid_flag=False, use_embeddings=False,
                                          compression=None, top_k=None, preserve_order=True):
    """Utility to run segmentation (and summarization if requested) and print results."""
    if not text:
        print("No text supplied.")
        return

    # Segment (with or without returning cnn probs)
    if do_summarize:
        sentences, cnn_probs = segment_text(text, model, use_hybrid_features=use_hybrid_flag, return_cnn_probs=True)
    else:
        sentences = segment_text(text, model, use_hybrid_features=use_hybrid_flag, return_cnn_probs=False)

    if not sentences:
        print("No sentences detected.")
        return

    # Print sentences
    print("\nSegmented Sentences:")
    for i, s in enumerate(sentences, 1):
        print(f"[{i}] {s}")

    # If summarization requested, run summarizer
    if do_summarize:
        print("\nInitializing summarizer...")
        # suppress the numpy.matrix TF-IDF warning from inside summarizer (it warns but proceeds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summarizer = SentenceSummarizer(use_embeddings=use_embeddings)

        print("\nComputing sentence weights (components: cnn_prob={:.2f}, textrank={:.2f}, tfidf={:.2f}, position={:.2f})...".format(
            summarizer.cnn_prob_weight, summarizer.textrank_weight,
            summarizer.tfidf_weight, summarizer.position_weight
        ))

        try:
            weights, component_scores = summarizer.compute_sentence_weights(
                sentences, text, cnn_probs if use_hybrid_flag else None
            )
        except Exception as e:
            warnings.warn(f"Summarizer weight computation failed: {e}. Using uniform weights.")
            weights = [1.0 / len(sentences)] * len(sentences)
            component_scores = None

        print("\nWeights (normalized):")
        for i, (sent, weight) in enumerate(zip(sentences, weights), 1):
            print(f"[{i}] {weight:.3f} ({weight*100:.1f}%)")

        # Generate summary
        selected_sentences, all_weights, _ = summarizer.summarize(
            sentences,
            text,
            cnn_probs if use_hybrid_flag else None,
            compression=compression,
            top_k=top_k,
            preserve_order=preserve_order
        )

        if top_k:
            summary_info = f"top {top_k} sentences"
        elif compression:
            summary_info = f"compression={compression}"
        else:
            summary_info = f"top {len(selected_sentences)} sentences (default)"

        print(f"\nSummary ({summary_info}):")
        for i, sent in enumerate(selected_sentences, 1):
            orig_idx = sentences.index(sent) + 1
            weight = weights[orig_idx - 1] if orig_idx - 1 < len(weights) else 0.0
            print(f"{i}) [{orig_idx}] {sent} — {weight*100:.1f}%")

    print("\n" + "="*50)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='LeXIDesk: Legal text sentence boundary detection and summarization'
    )
    parser.add_argument(
        '--summarize',
        action='store_true',
        help='Enable summarization mode (non-interactive)'
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
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive prompt mode (paste text, choose summarization)'
    )

    args = parser.parse_args()

    # Load models if not already loaded
    global baseline_crf_model, hybrid_crf_model, cnn_model, char_to_idx, device, vocab_size
    if baseline_crf_model is None:
        baseline_crf_model, hybrid_crf_model, cnn_model, char_to_idx, device, vocab_size = load_models()

    if hybrid_crf_model is None:
        print("Warning: Hybrid model not available. Using baseline model.")
        args.use_hybrid = False

    # Select model variable used for segmentation calls
    global model
    model = hybrid_crf_model if args.use_hybrid and hybrid_crf_model is not None else baseline_crf_model
    use_hybrid_flag = args.use_hybrid and hybrid_crf_model is not None and cnn_model is not None

    # Interactive mode
    if args.interactive:
        print("Interactive mode. Paste text (end with empty line). Press Ctrl+C to exit at any time.")
        try:
            while True:
                text = get_multiline_input("\nPaste legal text below (empty line to finish):")
                if not text:
                    print("No text entered. To exit interactive mode press Ctrl+C or enter nothing again.")
                    cont = input("Enter another? (y/n) [n]: ").strip().lower()
                    if cont == 'y':
                        continue
                    else:
                        break

                # Ask if user wants summarization for this input
                use_sum = None
                while use_sum is None:
                    choice = input("Run extractive summarization on this text? (y/n) [n]: ").strip().lower()
                    if choice == 'y':
                        use_sum = True
                    elif choice == 'n' or choice == '':
                        use_sum = False
                    else:
                        print("Please enter 'y' or 'n'.")

                # Ask whether to use hybrid model (if available)
                use_hybrid_for_this = use_hybrid_flag
                if hybrid_crf_model is not None:
                    uh_choice = input(f"Use hybrid CNN-CRF model for SBD? (y/n) [{'y' if use_hybrid_flag else 'n'}]: ").strip().lower()
                    if uh_choice == 'y' or (uh_choice == '' and use_hybrid_flag):
                        use_hybrid_for_this = True
                    else:
                        use_hybrid_for_this = False
                else:
                    use_hybrid_for_this = False

                # Run segmentation and optionally summarization
                run_segmentation_and_optional_summary(
                    text,
                    do_summarize=use_sum,
                    use_hybrid_flag=use_hybrid_for_this,
                    use_embeddings=args.use_embeddings,
                    compression=args.compression,
                    top_k=args.top_k,
                    preserve_order=args.preserve_order
                )

                again = input("Analyze another text? (y/n) [y]: ").strip().lower()
                if again != '' and again != 'y':
                    break

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
        return

    # Non-interactive modes (file / stdin / flags)
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
    elif args.summarize or not (args.text_file or args.stdin):
        # In CLI non-interactive use-case: if --summarize or no input sources, use default sample text
        text = """
The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review. This principle is outlined in § 1.3(a) of the legal code. The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. All proceedings were documented by the F.B.I. for review.
"""
    else:
        # Shouldn't reach here but guard anyway
        print("Error: Please provide --interactive, --text-file, or --stdin")
        sys.exit(1)

    text = text.strip()
    if not text:
        print("Error: No input text provided.")
        sys.exit(1)

    # If summarize flag set, run summarization; otherwise run segmentation-only display
    if args.summarize:
        run_segmentation_and_optional_summary(
            text,
            do_summarize=True,
            use_hybrid_flag=use_hybrid_flag,
            use_embeddings=args.use_embeddings,
            compression=args.compression,
            top_k=args.top_k,
            preserve_order=args.preserve_order
        )
    else:
        # segmentation-only
        run_segmentation_and_optional_summary(
            text,
            do_summarize=False,
            use_hybrid_flag=use_hybrid_flag,
            use_embeddings=args.use_embeddings,
            compression=None,
            top_k=None,
            preserve_order=True
        )


if __name__ == '__main__':
    main()
