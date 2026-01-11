
= Diffusion

Train network (DNN) to denoise images with different noise levels. In inference, begin with pure Gaussian noise and iteratively denoise → image matching training distribution.

*Key difference vs VAE/GAN:* VAE/GAN generate in one forward pass; diffusion does many refinement steps (can "correct itself").

Need two processes:
1. Fixed forward diffusion process q
2. Learned reverse denoising diffusion process $p_theta$

*Preprocessing:* normalize training images to 0 mean and unit variance

== Forward Diffusion q
- Starting point: starting image $x_0$
- Corrupt step by step over large number of steps T
- Sample noise from Gaussian distribution at each time step and add it to x
- T is large enough that image is indistinguishable from std Gaussian noise
- Final image: $x_T$

=== One Step Noising
Add small noise at each timestep $t=1..T$ with variance $beta_t$:
- Sample: $epsilon ~ N(0,1)$
- $x_t = sqrt(1 - beta_t) times x_(t-1) + sqrt(beta_t) times epsilon_(t-1)$
- Scaling ensures: if $x_(t-1)$ has mean 0 and var 1, then $x_t$ also stays mean 0, var 1

=== Reparameterization
Can sample $x_t$ at any arbitrary noise level conditioned on $x_0$ (sum of Gaussians is also Gaussian):
- Define $alpha_t = 1 - beta_t$ and $bar(alpha)_t = product_(i=1)^t alpha_i$
- Then sample directly from $x_0$: $x_t = sqrt(bar(alpha)_t) times x_0 + sqrt(1 - bar(alpha)_t) times epsilon$
- *Benefit:* training can pick random t without simulating all intermediate steps

=== Diffusion Schedules
$beta_t$ or equivalently $bar(alpha)_t$ are not constant and change with time. Schedule can be linear, quadratic, cosine etc.

*Linear:*
- $beta_t$ increases linear (e.g., 0.0001 → 0.2)
- Early: tiny noising steps → Late: larger steps (image already very noisy)

*Cosine:*
- Noise increases more gradually → often improves training efficiency and generation quality
- $bar(alpha)_t = cos^2(t/T times pi/2)$
- $x_t = cos(t/T times pi/2) times x_0 + sin(t/T times pi/2) times epsilon$

== Reverse Diffusion

=== Goal
Want $p(x_(t-1) | x_t)$ (denoise), but true distribution is intractable. Learn approximation with neural network (parameters $theta$).

=== What the NN Predicts (DDPM Training Simplification)
- Provide network with: noisy image $x_t$ + timestep (or schedule value)
- Network predicts noise $epsilon_theta (x_t, t)$
- Train with mean squared error: minimize $||epsilon - epsilon_theta (x_t, t)||^2$

== Training
- Take random sample $x_0$ from real unknown data distribution $q(x_0)$
- Sample noise level t where $t > 1 and t < T$ (random timestep)
- Sample noise from Gaussian distribution $epsilon ~ N(0,1)$
- Form $x_t$ using known schedule
- Train NN to predict $epsilon$ from $(x_t, t)$ (SGD on batches)

== U-Net
U-Net is encoder-decoder CNN with skip connections.

*Architecture:*
- *Encoder (downsampling):* Convolutions and pooling, extracts high-level global features
- *Decoder (upsampling):* Transposed convolutions / upsampling, reconstructs image details
- *Skip connections:* Copy encoder feature maps to decoder, preserve fine-grained spatial details

Helps DNN learn complex patterns and avoid vanishing gradient issues (provides highway for gradients to flow).

*Timestep encoding:* Use sinusoidal embedding to map scalar timestep/noise-level to higher-dim vector (like Transformers):
$ gamma(x) = (sin(2 pi e^0 f x), dots, sin(2 pi e^(L-1) f x), cos(2 pi e^0 f x), dots, cos(2 pi e^(L-1) f x)) $

== Generation (Sampling)
- Start with $x_T ~ N(0, I)$
- For $t = T .. 1$:
  - Predict noise (or mean)
  - Compute $x_(t-1)$ (denoise one step)
- Model predicts total noise component; iterative updates move from noisy → clean

== Latent Diffusion / Stable Diffusion / Imagen

=== Latent Diffusion Models (LDM)
*Key idea:* Run diffusion in latent space instead of pixel space:
- Autoencoder encodes image → latent
- Diffusion operates on latent (cheaper)
- Decoder reconstructs final image

=== Stable Diffusion (Key Points)
- Released Aug 2022 (public weights via Hugging Face)
- Denoising U-Net lighter because operates in latent space
- Autoencoder handles encoding/decoding "heavy lifting"
- Can be guided by text prompt via text encoder (v1 used CLIP; later versions use OpenCLIP)

=== Imagen (Pipeline Idea)
- Frozen text encoder: T5-XXL
- Text-to-image diffusion model (U-Net conditioned on text embeddings)
- Super-resolution diffusion upsamplers: 64×64 → 256×256 → 1024×1024 (still conditioned on text)

= Multi-Modality (DALL·E, Flamingo)

Train Generative Models to convert between two or more different kinds of data (e.g., text ↔ image, text ↔ video). Requires learning shared representation to bridge modalities.

== DALL·E 2

=== Architecture
1. *Encoder:* text → text embedding vector
   - Needs discrete text
   - Uses CLIP as text encoder

2. *Prior:* text embedding vector → image embedding vector
   - Two options: autoregressive model or diffusion model
   - Diffusion outperformed autoregressive model
   
   *Autoregressive Prior:*
   - Generates output sequentially, placing ordering on output tokens
   - Output of encoder fed to decoder, alongside current generated output image embedding
   - Output generated one element at a time, using teacher forcing to compare predicted next element to actual CLIP image embedding
   
   *Diffusion Prior:*
   - Diffusion process in embedded space
   - Noise image embedding → random noise
   - Learn to denoise step-by-step, conditioned on text embedding
   - Loss: average MSE across denoising steps

3. *Decoder:* generates image
   - Based on text prompt and predicted image embedding by prior
   - Final part of decoder is Upsampler (2 separate diffusion models)
   - First DF transforms image from 64×64 to 256×256
   - Second transforms from 256×256 to 1024×1024

=== Image Variations
- Compute CLIP image embedding for input image using CLIP image encoder
- Feed that embedding into decoder → generate variations

=== Limitations
- *Attribute binding limitation:* Model struggles to correctly associate attributes (e.g., color, position) with correct objects
- Confusion in relational prompts (e.g., "red cube on blue cube" vs "blue cube on red cube")
- Limited understanding of precise spatial relationships
- *Poor text rendering in images*
  - Reason: CLIP embeddings encode high-level semantics, not exact spelling or character-level details

== Flamingo (Vision Language Model)
VLM that can handle interleaved text + visual inputs (images + video frames).

=== Core Components (3)
- *Vision encoder* (frozen)
- *Perceiver Resampler*
- *Language model*

=== Video Handling (Slides' Recipe)
- Sample video at 1 frame/sec
- Run each frame through vision encoder → feature grids
- Add learned temporal encodings, flatten, concatenate

=== Perceiver Resampler (Why + How)
*Problem:* Images produce many spatial tokens (e.g., 14×14×1024) → too expensive for LM

*Solution:* Compress to fixed-size smaller set of visual tokens:
- Learn latent queries (few) that attend over all image tokens
- Cross-attention form: $Z = "softmax"(Q K^T) V$
  - Q = latent queries (small count)
  - K,V = image tokens
- Output: compact visual embeddings usable by language model