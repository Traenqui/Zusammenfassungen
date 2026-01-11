#import "../../template_zusammenf.typ": *
#import "@preview/wrap-it:0.1.1": wrap-content

= Diffusion Models

Train a DNN to *denoise* images at different noise levels.  
*Inference:* start from *pure Gaussian noise* and iteratively denoise → sample from the training distribution.

*Key difference vs VAE/GAN:* VAE/GAN generate in *one forward pass*; diffusion does *many refinement steps* (can “self-correct”).

*Two processes:*
- Fixed forward noising: $q$
- Learned reverse denoising: $p_theta$

*Preprocessing (common):* normalize training images to ~0 mean and ~unit variance.

== 1. Forward diffusion (noising) $q$
Goal: gradually corrupt clean data $x_0$ over $T$ steps until:
$ x_T approx cal(N)(0, I) $
(i.e., indistinguishable from standard Gaussian noise)

=== One-step noising
For $t = 1..T$ with schedule $beta_t$:
- Sample: $epsilon_(t-1) ~ cal(N)(0, I)$
- Update:
  $ x_t = sqrt(1 - beta_t) x_(t-1) + sqrt(beta_t) epsilon_(t-1) $

Scaling keeps mean ~0 and var ~1 if $x_(t-1)$ is normalized.

=== Reparameterization (sample any $t$ directly from $x_0$)
Define:
$ alpha_t = 1 - beta_t $  
$ bar(alpha)_t = product_(i=1)^t alpha_i $

Then:
$ x_t = sqrt(bar(alpha)_t) x_0 + sqrt(1 - bar(alpha)_t) epsilon, quad epsilon ~ cal(N)(0, I) $

*Benefit:* training can pick random $t$ without simulating all intermediate steps.

=== Diffusion schedules
Schedules define how $beta_t$ (or $bar(alpha)_t$) changes over time (linear/quadratic/cosine/...).

#table(
  columns: 2,
  table.header[Linear schedule][Cosine schedule (common)],
  [
    - $beta_t$ increases linearly (e.g. 0.0001 -> 0.2)
    - Early: tiny noising steps
    - Late: larger steps (image already noisy)
    - Example form:
      $ beta_t = beta_"min" + (t - 1)/(T - 1) (beta_"max" - beta_"min") $
  ],
  [
    - Noise increases more gradually → often better stability/quality
    - One form:
      $ bar(alpha)_t = cos^2(t/T pi/2) $
      $ x_t = cos(t/T pi/2) x_0 + sin(t/T pi/2) epsilon $
  ]
)

*Why Gaussian noise:* easy to sample, tractable math, known prior enables stable reverse denoising.

#hinweis("sum of Gaussians is also Gaussian")

== Reverse diffusion (denoising) $p_theta$
Goal: learn $p(x_(t-1) | x_t)$ (intractable exactly) using a neural network.

=== DDPM simplification: predict noise
Input: noisy sample $x_t$ + timestep $t$ (or noise-level embedding)  
Predict: noise $epsilon_theta(x_t, t)$

Loss (MSE):
$ min E[||epsilon - epsilon_theta(x_t, t)||^2] $

== Training loop (high level)
1. Sample clean data: $x_0 ~ q(x_0)$
2. Sample timestep: $t in {1, .., T}$ (often $1 < t < T$)
3. Sample noise: $epsilon ~ cal(N)(0, I)$
4. Create noisy input: $x_t$ using schedule (reparameterization)
5. Update network (SGD) to predict $epsilon$ from $(x_t, t)$

== Denoiser architecture: U-Net
U-Net = encoder–decoder CNN with skip connections.
- Encoder: downsample (conv + pooling) → global/context features
- Decoder: upsample (transposed conv / upsampling) → reconstruct details
- Skip connections: preserve spatial detail + improve gradient flow

*Timestep encoding:* embed scalar $t$ into a vector (often sinusoidal, transformer-style), e.g.
$ gamma(t) = (sin(2 pi e^0 f t), ..., sin(2 pi e^(L-1) f t), cos(2 pi e^0 f t), ..., cos(2 pi e^(L-1) f t)) $

== Sampling (generation)
Start: $x_T ~ cal(N)(0, I)$  
For $t = T, T-1, .., 1$:
- Predict noise (or mean)
- Compute $x_(t-1)$ (one denoising step)

Result: $x_0$ looks like data.  
Model predicts the noise component; repeated updates move noisy → clean.

== Latent diffusion and popular systems

=== Latent Diffusion Models (LDM)
Run diffusion in *latent space* instead of pixels:
- Autoencoder encodes image → latent
- Diffusion denoises latent (cheaper)
- Decoder reconstructs final image

=== Stable Diffusion (key points)
- Released Aug 2022 (public weights via Hugging Face)
- Denoising U-Net is lighter (latent space)
- Autoencoder does encoding/decoding “heavy lifting”
- Text guidance via text encoder (v1 used CLIP; later versions use OpenCLIP)

=== Imagen (pipeline idea)
- Frozen text encoder: T5-XXL
- Text-to-image diffusion (U-Net conditioned on text embeddings)
- Super-resolution diffusion upsamplers: 64x64 -> 256x256 -> 1024x1024 (still text-conditioned)

== Diffusion vs AE/VAE (why not the same)
- AE: bottleneck may lose detail; not timestep/noise-level conditioned
- VAE: KL regularization often yields blurrier reconstructions
- Diffusion: explicitly trains a *time-conditioned denoiser* across noise scales

*One-line takeaway:* add Gaussian noise until $x_T ~ cal(N)(0, I)$, then learn a time-conditioned network to reverse it.

= Multi-Modality (DALL·E 2, Flamingo)

== DALL·E 2
Pipeline:
1. Text encoder: text → text embedding (CLIP text encoder)
2. Prior: text embedding → image embedding
   - Autoregressive prior: sequential generation (teacher forcing vs CLIP image embedding)
   - Diffusion prior: denoise in embedding space conditioned on text; loss = average MSE across steps
3. Decoder: generate image conditioned on text + predicted image embedding
   - Diffusion upsamplers: 64x64 → 256x256 → 1024x1024

*Image variations:* CLIP image encoder → image embedding → decode variations.

*Limitations:*
- Attribute binding issues
- Confusion in relations (“red cube on blue cube” vs swapped)
- Weak precise spatial reasoning
- Poor text rendering (embeddings capture semantics, not exact spelling/characters)

== Flamingo (Vision-Language Model)
Handles interleaved text + images/video.

*Core components:*
- Frozen vision encoder
- Perceiver Resampler
- Language model

*Video handling (common recipe):*
- Sample video at ~1 fps
- Encode each frame → feature grids
- Add learned temporal encodings; flatten + concatenate

*Perceiver Resampler (why/how):*
- Problem: too many image tokens (e.g., 14x14x1024) for LM
- Solution: compress to fixed small set via cross-attention with latent queries:
  $ Z = "softmax"(Q K^T) V $
  where $Q$ = few learned latent queries, $K,V$ = image tokens → compact visual tokens for LM.