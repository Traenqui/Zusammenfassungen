#import "../../template_zusammenf.typ": *

= Vision Transformers (ViT)

== Why ViT?
- Transformers were highly successful in NLP → applied to vision.
- Naive self-attention over *pixels* is expensive:
  - sequence length would be $H times W$ (pixels)
  - attention cost scales ~ $O(n^2)$ in sequence length
- *Fix:* split image into *patches* and treat each patch as one token.

== ViT Architecture (Encoder-only)
- Split image into non-overlapping patches of size $P times P$.
- Flatten each patch and map it to a token embedding using a linear projection.
- Add positional encoding (usually learned in ViT) so the model knows patch order.
- Feed the token sequence into a Transformer *encoder*.
- Classification: use a pooled representation (often a special class token) → linear head → logits → probabilities.

=== Patches Math
Input image: $X in RR^(H times W times C) $

Flattened patches:
- Patch size: $P times P$
- $"#patches"$: $N = (H times W) / P^2 $
- Patch matrix: $X_p in RR^(N times (P^2 times C)) $

Token embeddings: $Z_0 = X_p W + b $
where $W in RR^((P^2 times C) times D)$ and $D$ is hidden dim.

=== ViT Flavors (Common Configurations)
#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([*Model*], [*Layers*], [*Hidden D*], [*MLP*], [*Heads*], [*Params*]),
  [ViT-Base], [12], [768], [3072], [12], [86M],
  [ViT-Large], [24], [1024], [4096], [16], [307M],
  [ViT-Huge], [32], [1280], [5120], [16], [632M],
)

=== ViT Advantages
- Scales well with model/data size (Transformer-style scaling).
- Self-attention can capture *global* relationships between distant patches.
- Strong performance with large-scale pretraining + transfer learning.

== ViT vs CNNs

=== Locality
*CNNs:*
- Strong locality inductive bias: nearby pixels/features are assumed related.
- Receptive field grows with depth; features built hierarchically.

*ViT:*
- No explicit locality bias: attention can connect any patch to any other patch immediately.
- With limited data, ViTs often need pretraining or hybrid designs to compete.

=== Translational Invariance
*CNNs:*
- Built-in translation equivariance/invariance (convolutions + pooling).
- Often generalize well with smaller datasets.

*ViT:*
- No built-in translation invariance.
- Typically needs more data / augmentation / pretraining to learn these invariances.

=== When to Use ViT vs CNN
*Limited data -> CNNs:*
- Inductive biases help generalization on small datasets.

*Real-time / low-latency -> CNNs:*
- Often faster and more compute-efficient on edge/mobile.

*Limited compute budget -> CNNs:*
- Typically fewer parameters / cheaper training (depending on setup).

*Long-range spatial dependencies -> ViTs:*
- Global self-attention connects distant regions naturally.

*Large-scale pretraining + transfer -> ViTs:*
- ViTs benefit strongly from large pretraining and can transfer well.

=== Mean Attention Distance (Definition)
For a single query patch/token $q$ in one attention head:
1. Consider all key tokens $k_i$.
2. Compute spatial distance $d_i$ between $q$ and each $k_i$ (e.g., Euclidean in image coordinates).
3. Weight distances by attention weights $a_i$:
   $ d(q) = sum_i a_i dot d_i $
4. Average $d(q)$ across all queries and examples → mean attention distance for that layer.
Interpretation: larger values indicate more global attention.

= Multi-Modal Transformers (CLIP)

CLIP learns a shared embedding space for images and text using many image-text pairs.

== Core Idea (Correct)
- Use *two separate encoders*:
  - image encoder (often ViT or CNN) -> image embedding $v_i$
  - text encoder (Transformer) -> text embedding $w_i$
- Train with a *contrastive objective* so matched pairs are close and mismatched pairs are far.
#hinweis[
Classic CLIP does *not* fuse image+text tokens via cross-attention.
Cross-attention fusion is typical in models like Flamingo/LLaVA-style systems, not standard CLIP.
]

== Contrastive Learning (CLIP-style)
For a batch of $N$ pairs:
- Normalize embeddings (often): $tilde(v)_i = v_i / norm(v_i) $, $tilde(w)_j = w_j / norm(w_j) $
- Similarity logits (with temperature $T$): $s_(i,j) = (tilde(v)_i dot tilde(w)_j) / T $

Symmetric loss (row + column classification):
- Image->Text: $L_"i2t" = - frac(1, N) sum_(i=1)^N log frac(exp(s_(i,i)), sum_(j=1)^N exp(s_(i,j))) $
- Text->Image: $L_"t2i" = - frac(1, N) sum_(i=1)^N log frac(exp(s_(i,i)), sum_(j=1)^N exp(s_(j,i))) $
Total: $L = frac(1, 2) (L_"i2t" + L_"t2i") $

*Intuition:* each image should assign highest probability to its matching text (and vice versa).

== SigLIP (Sigmoid Loss for Language–Image Pre-Training)
Instead of a softmax over the whole batch, SigLIP treats each pair as binary (match vs non-match):
- For each pair $(i,j)$:
  - label $y_(i,j)=1$ if $i=j$ else $0$
  - predict using $sigma(s_(i,j))$
- No global normalization across batch required.

== Zero-Shot Classification with CLIP
- Turn class labels into prompts (e.g., `"a photo of a [label]"`).
- Encode image and all prompt texts.
- Choose the label with maximum cosine similarity (or dot product) to the image embedding.