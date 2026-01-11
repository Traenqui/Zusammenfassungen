#import "../../template_zusammenf.typ": *

= LLMs (Large Language Models)

== Tokenizer
*Purpose:* Convert raw text into discrete token IDs for Transformer models.

*Pipeline:* Text → Tokens → Token IDs → Embeddings

*Tokenization levels:*
- Character-level: small vocab, long sequences
- Word-level: large vocab, OOV/unknown-word problem
- Subword-level (BPE, WordPiece, Unigram): frequent subwords, avoids OOV, good tradeoff

*Special tokens:* `<BOS>`/`<EOS>` boundaries, `<PAD>` batching, `<UNK>` unknown.

*Why it matters:* controls sequence length, vocab size, memory/compute, and behavior.

== Decoder-only Transformer (causal LM)
A *decoder-only* Transformer predicts the next token given previous ones:
- Training objective (autoregressive):
  $ L = - sum_(t=1)^T log p_theta(x_t | x_<t) $
- Uses *causal masking* so token $t$ cannot attend to future tokens $> t$.

#hinweis[
Decoder-only ≠ “has no attention”: it uses self-attention, but *masked* (causal).
]

== Training vs Inference
*Training (teacher forcing):*
- Model sees full prefix for each position (masked) and predicts all next tokens in parallel.
- Loss is summed/averaged over sequence positions.

*Inference (generation):*
- Generate token-by-token:
  - start with prompt
  - repeatedly sample next token from $p_theta(· | x_<t)$ and append
- Cannot parallelize across time steps (depends on previous outputs).

== KV Cache (fast inference)
During generation, self-attention reuses past keys/values:
- Without cache: recompute attention over all previous tokens each step.
- With *KV cache*: store $K_{1..t}, V_{1..t}$ and only compute new $K_t, V_t$.

Effect:
- Much faster decoding for long contexts
- Memory grows with sequence length ($"#tokens" times "#layers" times "#heads" times "head-dim"$)

== Encoder-only models (BERT, DistilBERT)

Encoder-only Transformers use bidirectional self-attention to produce contextual embeddings for
each token (good for understanding tasks, not autoregressive generation).

_BERT (Bidirectional Encoder Representations from Transformers)_: pretrained with _Masked Language Modeling (MLM)_ and (in the original paper) _Next Sentence Prediction (NSP)_. MLM: mask
tokens and predict them using left+right context.
_DistilBERT_: a smaller BERT-like encoder trained via knowledge distillation (student mimics
teacher). DistilBERT reduces depth (about half the layers) and uses distillation losses (incl. cosine
loss aligning hidden states) during training.

== Decoding / Sampling Strategies

=== Greedy decoding
Choose the most likely next token: $w_t = arg max_w P_t(w) $

Pros: deterministic, simple  
Cons: can be repetitive / low diversity

=== Temperature
Control randomness by scaling logits:
- Standard: $p = "softmax"(z / T)$
- $T < 1$: sharper (more deterministic)
- $T > 1$: flatter (more diverse)

$T = 1$ softmax unchanged, $T = 0$ argmax (greedy).

#hinweis[
The probability-power form $P(w)^(1/T)$ is equivalent only when $P$ came from a softmax over logits.
]

=== Top-k sampling
Sample only from the $k$ most probable tokens:
1. Take top-$k$ tokens by $P_t(w)$
2. Renormalize within this set
3. Sample

Pros: avoids very unlikely tokens  
Cons: fixed $k$ can be too strict or too loose depending on step

=== Nucleus sampling (Top-p)
Choose the smallest set whose cumulative probability mass reaches $p$.
1. Sort tokens by probability:
   $P_t(w_(1)) >= P_t(w_(2)) >= dots$
2. Find smallest $K$ such that:
   $sum_(i=1)^K P_t(w_(i)) >= p$
3. Nucleus set:
   $S_t = {w_(1), dots, w_(K)}$
4. Renormalize within $S_t$ and sample

Why nucleus may grow later:
- Later-step distributions can be less peaked → need more tokens to reach the same cumulative mass

=== Beam search
Keep the top-$k$ partial sequences (beams) by cumulative log-probability:
- expand each beam with candidate next tokens
- keep best $k$ overall, prune the rest

Pros: improves likelihood, common in seq2seq  
Cons: less diverse, can over-prefer generic high-probability sequences
