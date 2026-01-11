= Vision Transformers (ViT)

== Why ViT?
- Transformers successful in NLP → tried in images
- Naive self-attention would require each pixel to attend to every other pixel → quadratic costs
- *Fix:* create small patches of image and treat them as tokens

== ViT Architecture
- Split img into patches with feature extractor
- Build vocabulary of image patches with feature extractor
- Patches are input of transformer encoder-only model
- Model embeds input → produces raw logits that convert into final probabilities

=== Patches Math
Image split into n patches with feature extractor. Patches of equal dimensions represent words of sequence.

Input image: $X in RR^(H times W times C)$\
Sequence of patches: $X_p in RR^(N times (P^2 times C))$\
Patch size: $P times P$\
Number of patches: $N = (H times W) / P^2$ (effective sequence length)

=== Training ViT / ViT Flavors
#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([*Model*], [*Layers*], [*Hidden D*], [*MLP*], [*Heads*], [*Params*]),
  [ViT-Base], [12], [768], [3072], [12], [86M],
  [ViT-Large], [24], [1024], [4096], [16], [307M],
  [ViT-Huge], [32], [1280], [5120], [16], [632M],
)

=== ViT Advantages
- Inherits scaling capabilities of original Transformer model
- Can capture long-term dependencies better than CNN-only architectures
- Learn relationship between all patches in self-attention layers → more accurate predictions

== ViT vs CNNs

=== Locality
*CNNs:*
- Assume nearby parts in image are related
- Receptive Fields (RF) → size of region which defines feature

*ViT:*
- No assumption → RF is global
- Can see mean attention distance vs. depth
- Highly localized attention less pronounced in hybrid models that apply ResNet before Transformer

=== Translational Invariance
*CNNs:*
- Assume moved shape is same shape
- Invariance: can recognize entity in image, even when appearance or position varies
- Translation: each image pixel moved fixed amount in particular direction

*ViT:*
- No such assumption
- Needs large training dataset to compensate

=== When to Use ViT vs CNN
*Limited data → CNNs:* Strong inductive biases (locality, translation invariance) enable good generalization with small datasets. ViTs require large labeled datasets.

*Real-time / low-latency → CNNs:* Lower inference latency and higher computational efficiency, especially on mobile and edge devices.

*Limited compute budget → CNNs:* Fewer parameters, lower memory usage, cheaper training or fine-tuning.

*Long-range spatial dependencies → ViTs:* Self-attention captures global relationships between image patches.

*Pretrained models & transfer learning → ViTs:* Benefit strongly from large-scale pretraining and transfer learning.

=== Mean Attention Distance (Definition)
For single query pixel q:
1. Consider all key pixels $k_i$ in patch
2. Compute spatial distance between q and each $k_i$ (e.g., Euclidean). Call it $d_i$
3. Multiply each distance by attention weight $a_i$ that q assigns to $k_i$
4. Weighted distance for q: $sum_i a_i dot d_i$
5. Average across all queries in patch or image
6. Average across multiple images (e.g., 128 images) to get mean attention distance for layer

== Multi-Modal Transformers (CLIP)
Train on many image-text pairs; map both modalities into shared embedding space.

- Feature extractor like ViT produces image tokens
- Text is also input token
- Attention layer learns relationships between image and text tokens with cross attention
- Output is raw logits

=== Contrastive Learning
Encode inputs (e.g., image and text) into vectors:
1. Compute similarity (cosine similarity or dot product) between all pairs in batch
2. Apply contrastive loss (e.g., InfoNCE) that:
   - Maximizes similarity for matching pairs
   - Minimizes similarity for non-matching pairs

$ -1/N sum_i ln (e^(v_i dot w_i \/ T)) / (sum_j e^(v_i dot w_j \/ T)) - 1/N sum_j ln (e^(v_j dot w_j \/ T)) / (sum_i e^(v_i dot w_j \/ T)) $

For $v_i$, take dot product with all targets $w_i$ to get similarities. Apply softmax over similarities. Compute cross-entropy loss treating correct match $w_i$ as true class.

*Intuition:* $v_i$ should assign most probability to its positive pair $w_i$.

=== SigLip (Sigmoid Loss for Language Image Pre-Training)
Instead of softmax + cross-entropy over all negative pairs in batch, SigLIP uses binary classification (sigmoid) for each pair.

- For each image-text pair $(v_i, w_j)$ compute dot product and apply sigmoid to classify positive (i=j) or negative (i≠j)
- No global normalization across batch required

=== Zero-Shot Classification with CLIP
- Turn labels into text prompts (e.g., "a photo of a [label]")
- Encode image and all label texts; choose label with max cosine similarity