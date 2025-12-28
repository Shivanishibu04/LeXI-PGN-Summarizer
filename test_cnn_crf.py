"""
Test script to verify CNN-CRF integration for sentence segmentation.
"""

import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils.preprocessing import segment_sentences, HAS_CNN_CRF

print("="*80)
print("CNN-CRF INTEGRATION TEST")
print("="*80)

print(f"\n✓ CNN-CRF Model Available: {HAS_CNN_CRF}")

test_text = """
The court's decision in Marbury v. Madison, 5 U.S. 137 (1803), established the principle of judicial review. This principle is outlined in § 1.3(a) of the legal code. The defendant, Mr. Smith, was subsequently charged under 18 U.S.C. § 1001. All proceedings were documented by the F.B.I. for review.
"""

print(f"\n\nTest Text:\n{'-'*80}\n{test_text.strip()}\n{'-'*80}")

print("\n\nSegmenting with CNN-CRF model...")
sentences = segment_sentences(test_text, use_cnn_crf=True)

print(f"\n✓ Detected {len(sentences)} sentences:")
print("-"*80)
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
