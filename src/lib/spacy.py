import os
import spacy

dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(dir, "../data/models/en_core_web_sm"))
if not os.path.exists(model_path) or not os.path.isdir(model_path):
    print(f"Downloading spaCy model to {model_path}")
    os.makedirs(model_path)
    os.system(f"python -m spacy download en_core_web_sm --direct --destination {model_path}")
nlp = spacy.load(model_path)

def clamp(value, minimum, maximum):
    return max(min(value, maximum), minimum)

def word_window(words, max_words=6, min_words=3):
    if max_words <= 0:
        raise ValueError("Window size must be a positive integer.")
    if len(words) < min_words:
        raise ValueError("Sequence length must be greater than or equal to the window size.")
    i = 0
    while i < len(words):
        size = 1
        for idx, word in enumerate(words[i:i + max_words]):
            should_break = False
            if idx != 0:
                for breakable_prefix in ["(", "\""]:
                    if not f"{word}".startswith(breakable_prefix): continue
                    should_break = True
                    if idx + 1 < min_words: should_break = False
                    break
            for breakable_suffix in [",", ";", ":", "(", ")", "\"", "-"]:
                if not f"{word}".endswith(breakable_suffix): continue
                size = idx + 1
                should_break = True
                if idx + 1 < min_words: should_break = False
                break
            if should_break:
                break
            size = idx + 1
        remaining = len(words) - i + 1
        # Clamp size between (least of min_words or remaining) and max_words
        size = clamp(size, clamp(min_words, 1, remaining), max_words)
        yield words[i:i + size]
        i += size

def chunk_words(text, max_words=10, min_words=4):
    doc = nlp(text)

    chunks = []

    for sent in doc.sents:
        ws_words = [token.text_with_ws for token in sent]
        words = [word.strip() for word in ''.join(ws_words).split()]
        if not len(words): continue
        for window in word_window(words, clamp(max_words, len(words), max_words), clamp(min_words, 1, len(words))):
            chunks.append(' '.join(window).replace("&", "and"))

    return chunks
