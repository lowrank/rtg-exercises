"""
Language Entropy Estimation via n-Gram Perplexity
===================================================
Computes cross-entropy and perplexity of n-gram models.
Supports two corpora: WikiText-2 and the Brown Corpus.

Usage:
  python3 perplexity.py             # default: WikiText-2
  python3 perplexity.py brown       # Brown Corpus
"""
from collections import Counter, defaultdict
import math, re, sys, io, zipfile, urllib.request


def tokenize(text):
    tokens = re.findall(r"[a-z]+|[.!?]", text.lower())
    # Insert <s> at sentence boundaries (. ! ?)
    result = ['<s>']  # start of first sentence
    for t in tokens:
        result.append(t)
        if t in '.!?':
            result.append('<s>')
    return result


def load_wikitext2():
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    try:
        with urllib.request.urlopen(url) as f:
            return f.read().decode('utf-8')
    except Exception:
        print("Cannot download WikiText-2. Using synthetic text.")
        return "the cat sat on the mat . the dog sat on the log . " * 500


def load_brown():
    """Download and load the Brown Corpus (1M words, 1961 American English)."""
    url = "https://raw.githubusercontent.com/nltk/nltk_data/refs/heads/gh-pages/packages/corpora/brown.zip"
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        text = ""
        for name in zf.namelist():
            if name.startswith('brown/') and not name.endswith('/'):
                try:
                    text += zf.read(name).decode('latin-1') + " "
                except:
                    pass
        return text
    except Exception as e:
        print(f"Cannot download Brown Corpus: {e}")
        print("Using synthetic text.")
        return "the cat sat on the mat . the dog sat on the log . " * 500


def build_ngram_model(tokens, n=2, vocab=None):
    if vocab is None:
        vocab = set(tokens)
    counts = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        ctx = tuple(tokens[i:i+n-1]) if n > 1 else ()
        nxt = tokens[i + n - 1]
        counts[ctx][nxt] += 1
    return counts, vocab


ALPHA = 0.001

def ngram_prob(ctx, nxt, counts, vocab, n, alpha=ALPHA):
    ctx_counts = counts.get(ctx, Counter())
    total = sum(ctx_counts.values()) + alpha * len(vocab)
    return (ctx_counts.get(nxt, 0) + alpha) / total


def compute_perplexity(test_tokens, counts, vocab, n, alpha=ALPHA):
    log_prob = 0.0; N = 0
    i = 0
    while i < len(test_tokens):
        nxt = test_tokens[i]
        if nxt == '<s>':
            i += 1  # skip, reset context implicitly
            continue
        ctx = tuple(test_tokens[max(0,i-n+1):i]) if n > 1 else ()
        # Skip context tokens that are <s>
        ctx = tuple(t for t in ctx if t != '<s>')
        if len(ctx) < n-1:
            i += 1; continue  # not enough context yet
        p = ngram_prob(ctx, nxt, counts, vocab, n, alpha)
        log_prob += math.log2(p); N += 1; i += 1
    ce = -log_prob / N
    return ce, 2 ** ce


# ═══════════════════════════════════════════════════════════════
# Class-based trigram (Brown et al. 1992 style)
# ═══════════════════════════════════════════════════════════════

def build_class_model(train, n_classes=100, freq_cutoff=200):
    """
    Cluster words into classes by frequency.
    - Top `freq_cutoff` words each get their own class.
    - Remaining words are grouped into `n_classes` equal-frequency bins.
    Returns: word→class mapping, class→word list, class n-gram counts.
    """
    freq = Counter(train)
    sorted_words = [w for w, _ in freq.most_common()]
    word_to_class = {}
    class_to_words = defaultdict(list)
    cid = 0
    for i, w in enumerate(sorted_words):
        if i < freq_cutoff:
            word_to_class[w] = cid
            class_to_words[cid].append(w)
            cid += 1
    # Remaining words into bins
    remaining = [w for w in sorted_words if w not in word_to_class]
    bin_size = max(1, len(remaining) // n_classes)
    for i, w in enumerate(remaining):
        cid = freq_cutoff + i // bin_size
        word_to_class[w] = cid
        class_to_words[cid].append(w)
    return word_to_class, class_to_words


def class_trigram_perplexity(train, test, word_to_class, class_to_words, alpha=ALPHA):
    """
    P(w_t | w_{t-2}, w_{t-1}) = P(w_t | class(w_t)) * P(class(w_t) | class(w_{t-2}), class(w_{t-1}))

    Follows Brown et al. (1992). Uses add-alpha smoothing on both components.
    """
    # Assign an UNK class for OOV words (one past the max class id)
    unk_class = max(word_to_class.values()) + 1
    word_to_class = dict(word_to_class)  # don't mutate caller's dict

    # Map tokens to classes, filtering <s> for n-gram construction
    train_cls, train_words = [], []
    for t in train:
        if t == '<s>':
            continue  # sentence boundary, not part of n-gram stream
        cid = word_to_class.get(t, unk_class)
        train_cls.append(cid)
        train_words.append(t)

    test_cls, test_words = [], []
    for t in test:
        if t == '<s>':
            continue
        cid = word_to_class.get(t, unk_class)
        test_cls.append(cid)
        test_words.append(t)

    n_classes = len(class_to_words) + 1  # +1 for UNK class

    # P(w | class): MLE with add-alpha smoothing
    word_counts = Counter(train_words)
    class_counts = Counter()
    for w, cid in zip(train_words, train_cls):
        class_counts[cid] += 1

    word_given_class = {}
    for cid, words in class_to_words.items():
        total = class_counts.get(cid, 0)
        nw = len(words)
        for w in words:
            word_given_class[w] = (word_counts.get(w, 0) + alpha) / (total + alpha * nw)
    # UNK class fallback
    V = len(set(train_words))
    p_unk_word = 1.0 / V  # uniform over training vocab for unseen words

    # Class trigram: P(c_t | c_{t-2}, c_{t-1})
    cls_counts = defaultdict(Counter)
    for i in range(2, len(train_cls)):
        ctx = (train_cls[i-2], train_cls[i-1])
        cls_counts[ctx][train_cls[i]] += 1
    cls_totals = {ctx: sum(c.values()) for ctx, c in cls_counts.items()}

    logp = 0.0; N = 0
    for i in range(2, len(test_cls)):
        t = test_words[i]; c = test_cls[i]
        ctx = (test_cls[i-2], test_cls[i-1])

        # P(c_t | c_{t-2}, c_{t-1}) with add-alpha smoothing
        total_c = cls_totals.get(ctx, 0) + alpha * n_classes
        p_c = (cls_counts.get(ctx, Counter()).get(c, 0) + alpha) / total_c

        # P(w_t | c_t)
        p_w_given_c = word_given_class.get(t, p_unk_word)

        logp += math.log2(p_c * p_w_given_c)
        N += 1

    ce = -logp / N
    return ce, 2**ce


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "wikitext"
    if corpus.lower() == "brown":
        name = "Brown Corpus"
        text = load_brown()
    else:
        name = "WikiText-2"
        text = load_wikitext2()

    tokens = tokenize(text)
    print(f"Corpus: {name} — {len(tokens):,} tokens, vocab: {len(set(tokens)):,}")

    split = int(0.8 * len(tokens))
    train, test = tokens[:split], tokens[split:]
    vocab = set(train)
    V = len(vocab)

    uniform_ppl = V
    uniform_bits = math.log2(V)
    print(f"\n{'Model':<18} {'Bits/token':<14} {'PPL':<12}")
    print("-" * 44)
    print(f"{'Uniform':<18} {uniform_bits:<14.2f} {uniform_ppl:<12,.0f}")

    alpha = ALPHA
    for n in [1, 2, 3]:
        counts, _ = build_ngram_model(train, n=n, vocab=vocab)
        ce, ppl = compute_perplexity(test, counts, vocab, n, alpha=alpha)
        print(f"{f'{n}-gram':<18} {ce:<14.2f} {ppl:<12,.1f}")

    # Class-based trigram
    try:
        word_to_class, class_to_words = build_class_model(train, n_classes=200, freq_cutoff=300)
        ce, ppl = class_trigram_perplexity(train, test, word_to_class, class_to_words, alpha=alpha)
        n_cls = len(class_to_words) + 1  # +1 for UNK class
        print(f"{'Class trigram':<18} {ce:<14.2f} {ppl:<12,.1f}  ({n_cls} classes)")
    except Exception as e:
        print(f"Class-based model failed: {e}")

    print(f"\nBrown et al. (1992): 7.95 bits/word, PPL ≈ 247 (class-based trigram, ~1000 classes, MI clustering)")
    print(f"Our class model: frequency-based bucketing, add-{alpha} smoothing — a simplified version.")


if __name__ == '__main__':
    main()
