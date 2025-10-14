# In src/feature_extractor.py (REPLACE THE ENTIRE FILE)

import re

def get_token_signature(token):
    # (This function remains the same)
    sig = []
    for char in token:
        if char.islower():
            sig.append('c')
        elif char.isupper():
            sig.append('C')
        elif char.isdigit():
            sig.append('D')
        else:
            sig.append(char)
    return "".join(sig)

def get_token_length_feature(token):
    # (This function remains the same)
    length = len(token)
    if length < 4:
        return 'short'
    elif length <= 6:
        return 'normal'
    else:
        return 'long'

def token_to_features(token, text, start, end):
    """
    Extracts features for a single token, now including context from the original text.
    """
    # Character before the token
    char_before = text[start - 1] if start > 0 else '<BOS>' # Beginning of String
    # Character after the token
    char_after = text[end] if end < len(text) else '<EOS>' # End of String
    
    return {
        'token': token,
        'lower': token.lower(),
        'sig': get_token_signature(token),
        'len_cat': get_token_length_feature(token),
        'is_lower': token.islower(),
        'is_upper': token.isupper(),
        'is_title': token.istitle(),
        'is_digit': token.isdigit(),
        'char_before': char_before,
        'is_space_before': char_before.isspace(),
        'char_after': char_after,
        'is_space_after': char_after.isspace(),
    }

def add_neighboring_token_features(sentence_features):
    """
    Adds features from neighboring tokens to each token's feature dict.
    CRF models work best when they know about their immediate context.
    """
    for i in range(len(sentence_features)):
        # Feature for the previous token
        if i > 0:
            prev_features = sentence_features[i-1]
            sentence_features[i]['prev_token'] = prev_features['token']
            sentence_features[i]['prev_lower'] = prev_features['lower']
            sentence_features[i]['prev_is_title'] = prev_features['is_title']
            sentence_features[i]['prev_is_upper'] = prev_features['is_upper']
        else:
            # Mark as beginning of sentence
            sentence_features[i]['BOS'] = True

        # Feature for the next token
        if i < len(sentence_features) - 1:
            next_features = sentence_features[i+1]
            sentence_features[i]['next_token'] = next_features['token']
            sentence_features[i]['next_lower'] = next_features['lower']
            sentence_features[i]['next_is_title'] = next_features['is_title']
            sentence_features[i]['next_is_upper'] = next_features['is_upper']
        else:
            # Mark as end of sentence
            sentence_features[i]['EOS'] = True
    return sentence_features