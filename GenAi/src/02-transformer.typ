#import "../../template_zusammenf.typ": *

= Transformers

A _transformer_ uses attention to process sequences in parallel (unlike RNNs which are sequential).

== Why Not Increase N (in N-grams)?
- Context space explodes: larger $n$ ⇒ many more possible $n$-grams (data sparsity) → overfitting
- Data needed grows quickly: long contexts require far more examples to estimate reliably
- Fixed window is inflexible: long-range dependencies exist; fixed windows waste capacity
- *Attention solves it:* dynamically focuses on relevant tokens anywhere in the context

== Flavors of Transformers
- _Encoder-only_ (e.g., BERT): understanding tasks (classification, retrieval/embeddings, QA)
- _Decoder-only_ (e.g., GPT): generation (causal LM / autoregressive)
- _Encoder–decoder_ (e.g., T5): seq2seq (translation, summarization)

== Tokens, Embeddings, Positional Encoding
- Tokens $t_1, ..., t_n$ (from a tokenizer) → token embeddings
- Token embedding matrix: $E in RR^(|V| times d_"model")$
- Positional encodings add order info:

$ p_i = cases(sin("pos" / 10000^((2k) / d_"model")) "if pos is even", cos("pos" / 10000^((2k) / d_"model")) "else") $
- Input vectors: $x_i = E[t_i] + p_i$
  - $p_i$ is positional information (either *fixed* sinusoidal or *learned* position embeddings)

#hinweis[
If positional encodings are *fixed* (sin/cos), they add *0 trainable parameters*.
If they are *learned*, they add $n_"max" times d_"model"$ parameters.
]

== Self-Attention (Scaled Dot-Product)
The position-aware embedding $x_i$ is projected into (Query, Key, Value):

_Q_ (*queries*): $Q = X W_Q$ "What am I looking for?"\
_K_ (*keys*): $K = X W_K$ "What do I offer?"\
_V_ (*values*): $V = X W_V$ "The content to transfer"

$W_Q, W_K, W_V$ are learned and different. Positional information is contained in Q, K, and V.

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) dot V $

Here $Q K^T in RR^(n times n)$ are similarity scores; dividing by $sqrt(d_k)$ stabilizes training; each row of softmax sums to 1.

Typical choice: $d_k = d_"model" / h$.

== Feed-Forward Network (FFN)
A position-wise neural
network applied independently to each token, adding non-linearity and expressive power afterattention.
Applied independently per token:
$ "FFN"(x) = W_2 "relu"(W_1 x + b_1) + b_2 $

Dimensions:
- $W_1 in RR^(d_"model" times d_"ff")$, $W_2 in RR^(d_"ff" times d_"model")$

== Cross-Attention (Encoder–Decoder)
Used in encoder–decoder models:
- Decoder provides queries $Q$ (from decoder states)
- Encoder provides keys/values $K,V$ (from encoder outputs)

$ "CrossAttn" = "Attn"(Q_"dec", K_"enc", V_"enc") $


_Causal Masking_: (Decoder-only) Masking prevents the attention mechanism from attending to certain positions, e.g. padding tokens or future tokens in autoregressive decoding.
_Multi-Head Attention_: Multiple attention heads run in parallel, each with its own Q, K, V projections, allowing the model to capture different types of relationships simultaneously.
_Layer Normalization_: Normalizes activations across features for each token, stabilizing training and improving convergence in deep Transformer models.
_Batch Normalization_: Normalizes activations across the batch dimension, but is less suitable for Transformers due to variable sequence lengths. 
_Dropout_: Randomly disables neurons during training to reduce overfitting and
improve generalization.

== Example: Transformer Parameter Count

Vocabulary: $|V| = 100$, model dim: $d_"model" = 32$, heads: $h = 4$ → $d_"head" = 8$, FFN dim: $d_"ff" = 64$, Layers: $L = 2$, Seq length: $n=10$

*Token Embeddings:* $V times d_"model" = 100 times 32 = 3200$\
*Positional Embeddings:* $n times d_"model" = 10 times 32 = 320$\
*Attention params:* $W_Q, W_K, W_V => 3(32 times 32) = 3072$; $W_O: 32 times 32 = 1024$\
*Total attention per layer:* $3072 + 1024 = 4096$\
*FFN per layer:* $2(32 times 64) + 64 + 32 = 4192$\
*LayerNorm per layer:* $2 times 2 times 32 = 128$\
*Total per layer:* $4096 + 4192 + 128 = 8416$\
*Overall:* $2 times 8416 + 3200 + 320 = 20352$

#hinweis[
Decoder-only LMs often have an output projection to vocab (sometimes weight-tied to token embeddings). If untied, add $d_"model" times |V|$ (+ bias).
]
