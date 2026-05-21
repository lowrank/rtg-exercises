# Language Entropy Estimation via $n$-Gram Perplexity



## 1. Objective

This experiment estimates the **entropy of English** by building simple $n$-gram language models and measuring their **perplexity** on held-out text. 

---

## 2. Background

### Perplexity = Effective Branching Factor

A language model assigns a probability $P(w_t \mid \text{context})$ to each token. The **cross-entropy** (log scale) and **perplexity** (linear scale) are:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log_2 P(w_t \mid w_{<t}) \quad \text{(bits per token)}$$

and 

$$\text{PPL} = 2^{\mathcal{L}}$$

### Landmark Estimates of English Entropy

| Paper | Method | Bits/char | Bits/word | PPL |
|---|---|---|---|---|
| Shannon (1951) | Human letter-guessing | 0.6–1.3 | — | 8–64 |
| Cover \& King (1978) | Gambling | ~1.25 | — | — |
| Brown et al. (1992) | Trigram on Brown Corpus | 1.75 | 7.95 | ~247 |
| GPT-2 (2019) | Transformer | ~1.0 | — | ~18 |
| Llama-3 (2024) | Transformer | ~0.8 | — | ~5 |

---

## 3. The $n$-Gram Model

An $n$-gram model approximates the probability of a token given the previous $n-1$ tokens:

$$P(w_t \mid w_{t-1}, \dots, w_{t-n+1}) = \frac{\text{count}(w_{t-n+1}, \dots, w_t) + \alpha}{\text{count}(w_{t-n+1}, \dots, w_{t-1}) + \alpha V}$$

The $+\alpha$ terms are **add-$\alpha$ smoothing** ($\alpha = 0.001$ for example). Without it, any unseen $n$-gram would have probability zero (infinite perplexity).

- **$n=1$ (unigram)**: $P(w_t)$ — no context, just word frequency
- **$n=2$ (bigram)**: $P(w_t \mid w_{t-1})$ — one word of context
- **$n=3$ (trigram)**: $P(w_t \mid w_{t-2}, w_{t-1})$ — two words of context

---

## 4. Class-Based Trigram Model (Brown et al., 1992)

### 4.1 Probability Model

Assume the probability is modeled by the following form:

$$
\begin{aligned}
P(w_t \mid w_{t-2}, w_{t-1})
&= P_{1}\bigl(w_t \mid c(w_t)\bigr) \cdot P_2\bigl(c(w_t) \mid c(w_{t-2}),\, c(w_{t-1})\bigr)
\end{aligned}
$$

### 4.2 Normalization Check

$$
\begin{aligned}
\sum_{w} P(w \mid w_{t-2}, w_{t-1})
&= \sum_{c=1}^C P_2(c\mid c_{t-2}, c_{t-1})
\underbrace{\sum_{w \in c} P_1(w_t \mid c)}_{=\,1} \\[4pt]
&= \sum_{c=1}^{C} P_2(c \mid \ldots) \;=\; 1
\end{aligned}
$$

This probability model significantly simplifies the computation. 

### 4.3 Cross-Entropy and PPL

For each test token $w_t$ (skipping $\langle s \rangle$ sentence boundaries):

$$
\log P(w_t \mid w_{t-2}, w_{t-1})
= \underbrace{\log P_1(w_t \mid c_t)}_{\text{word from class}}
\;+\; \underbrace{\log P_2(c_t \mid c_{t-2}, c_{t-1})}_{\text{class from context}}
$$

$$
\text{CE} = -\frac{1}{N}\sum \log P, \qquad \text{PPL} = 2^{\text{CE}}
$$

---

## 5. Code

The file `perplexity.py` provides:

| Function | Purpose |
|---|---|
| `tokenize(text)` | Split text into lowercase word tokens |
| `load_wikitext2()` | Download WikiText-2 or use synthetic text |
| `load_brown()` | Download and parse the Brown Corpus |
| `build_ngram_model(tokens, n)` | Build $n$-gram counts (smoothing applied at query time) |
| `ngram_prob(ctx, nxt, counts, vocab, n)` | Compute $P(\text{nxt} \mid \text{ctx})$ |
| `compute_perplexity(test, counts, vocab, n)` | Compute cross-entropy and PPL on test data |
| `build_class_model(train, n_classes, freq_cutoff)` | Build frequency-based class mapping |
| `class_trigram_perplexity(train, test, w2c, c2w)` | Compute PPL using class-based trigram decomposition |

Run:
```bash
python3 perplexity.py              # WikiText-2
python3 perplexity.py brown        # Brown Corpus
```

---

## 6. References

1. Shannon, C.E. (1951). Prediction and Entropy of Printed English. *Bell System Technical Journal*, 30, 50–64.
2. Cover, T.M. \& King, R.C. (1978). A Convergent Gambling Estimate of the Entropy of English. *IEEE Trans. Info. Theory*, 24(4), 413–421.
3. Brown, P.F. et al. (1992). Class-Based $n$-gram Models of Natural Language. *Computational Linguistics*, 18(4), 467–479.
4. Brown, P.F. et al. (1992). An Estimate of an Upper Bound for the Entropy of English. *Computational Linguistics*, 18(1), 31–40.
5. Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. (GPT-2)
6. Touvron, H. et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.
