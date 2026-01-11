#import "../../template_zusammenf.typ": *

= Transformers

A _transformer_ uses attention mechanisms to process sequences in parallel, unlike RNNs which process sequentially.

== Why Not Increase N (in N-grams)?
- Scaling N explodes context space: counts blow up; neural inputs and params grow and overfit
- Data needed grows fast: more context → more examples needed
- Fixed window inflexible: long sparse dependencies are common; fixed windows waste capacity
- *Attention solves it:* dynamically focuses on relevant tokens, better efficiency and flexibility

== Flavors of Transformers
- _Encoder-Only_ (e.g., BERT): Good for understanding tasks like classification, QA (Embedding Models)
- _Decoder-Only_ (e.g., GPT): Good for generation tasks (Causal LM / autoregressive)
- _Encoder-Decoder_ (e.g., T5): Good for seq2seq tasks like translation

== Inputs: Tokens, Embeddings, Positional Encoding

_Tokenization_: Text split into tokens #hinweis[(subwords)], mapped to integer IDs\
_Embedding_: Matrix $E$ maps token IDs to vectors in $d_"model"$ dimensions\
_Positional Encoding_: Explicit position information added to embeddings

Transformers are order-invariant, so positional information is added:
$ x_i = "TokenEmbedding"_i + "PositionalEmbedding"_i $

Positions encoded using sine and cosine functions:
$ "PE"("pos", 2k) = sin("pos" / 10000^((2k) / d_"model")), "PE"("pos", 2k + 1) = cos("pos" / 10000^((2k) / d_"model")) $

Low frequencies encode coarse position, high frequencies encode fine-grained position. These embeddings are fixed (not learned).

Final input vectors: $X = "Embedding"("tokens") + "PE"$ with $X in RR^("seq_len" times d_"model")$

== Self-Attention

The position-aware embedding $x_i$ is projected into (Query, Key, Value):

_Q_ (*queries*): $Q = X W_Q$ "What am I looking for?"\
_K_ (*keys*): $K = X W_K$ "What do I offer?"\
_V_ (*values*): $V = X W_V$ "The content to transfer"

$W_Q, W_K, W_V$ are learned and different. Positional information is contained in Q, K, and V.

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) dot V $

Here $Q K^T in RR^(n times n)$ are similarity scores; dividing by $sqrt(d_k)$ stabilizes training; each row of softmax sums to 1.

*Masking:* Prevents attention to certain positions (e.g., padding tokens or future tokens in autoregressive decoding).

== Multi-Head Attention

Multiple attention heads run in parallel, each with own $W_Q^((h)), W_K^((h)), W_V^((h))$ projections, allowing model to capture different relationships simultaneously.

$ Y_h = "Attention"(Q_h, K_h, V_h), quad "MHA"(X) = "Concat"(Y_1, dots, Y_H) W_O $

== Feed-Forward Layer (FFN)

Position-wise neural network applied independently to each token, adding non-linearity and expressive power after attention:

$ "FFN"(x) = W_2 ("ReLU"(W_1 x + b_1)) + b_2 $

Parameters per layer:
$ "FFN params" = 2 times d_"model" times d_"ff" + d_"ff" + d_"model" $
where $d_"ff"$ is typically $approx 4 d_"model"$

== Layer Normalization

Normalizes activations across features for each token, stabilizing training and improving convergence:

$ mu = 1/d sum^d_(j=1) x_j, quad sigma^2 = 1/d sum^d_(j=1)(x_j - mu)^2 $
$ "LN"(x) = gamma ⊙ (x - mu)/(sqrt(sigma^2 + epsilon)) + beta $

Where $gamma$ and $beta$ are learnable parameters.

*Batch Normalization:* Normalizes activations across the batch dimension, less suitable for Transformers due to variable sequence lengths.

*Dropout:* Randomly disables neurons during training to reduce overfitting.

== Cross-Attention

In encoder-decoder models:
Encoder output $H in RR^(n_"src" times d)$, decoder states $D in RR^(n_"tgt" times d)$.

$ "CrossAttention"(D, H) = "Attention"(D W_Q, H W_K, H W_V) $

Decoder can attend to most relevant source tokens while generating each target token.

== Causal Masking

Prevents seeing future tokens. Apply triangular mask so position $i$ can only attend to positions $<= i$. Scores for $j > i$ set to $-infinity$ before softmax, making attention weight $0$.

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