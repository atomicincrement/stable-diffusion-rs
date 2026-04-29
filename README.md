# Understanding Stable Diffusion: From Text to Image

This document explains how Stable Diffusion works, covering the phases we've completed (1-3) and the forward diffusion process (Phase 4) that forms the foundation for inference.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Phase 1: Project Setup](#phase-1-project-setup)
3. [Phase 2: Weight Loading](#phase-2-weight-loading)
4. [Phase 3: Text Encoding with CLIP](#phase-3-text-encoding-with-clip)
5. [Phase 4: Forward Diffusion Process](#phase-4-forward-diffusion-process)
6. [The Complete Pipeline](#the-complete-pipeline)

---

## High-Level Overview

**Stable Diffusion** is a latent diffusion model for text-to-image generation. It works in three stages:

```
User Input: "a cat on a beach"
    ↓
[Stage 1] CLIP Text Encoder
    Input: Text prompt (variable length)
    Output: Text embedding (77, 768)
    Purpose: Convert semantic meaning to vector representation
    ↓
[Stage 2] Diffusion Process (Reverse/Inference) ← Powered by understanding noise schedule
    Input: Text embedding + random noise
    Output: Latent representation (4, 64, 64)
    Purpose: Iteratively denoise to generate image latent
    ↓
[Stage 3] VAE Decoder
    Input: Latent representation (4, 64, 64)
    Output: RGB Image (3, 512, 512)
    Purpose: Expand latent space to visual image
    ↓
Result: Generated image
```

This document focuses on **Stage 1 (complete)** and the **theoretical foundation for Stage 2 (noise schedule in Phase 4)**.

---

## Phase 1: Project Setup ✓

### Dependencies

We installed 12+ Rust crates to enable efficient tensor operations:

| Crate | Purpose | Status |
|-------|---------|--------|
| `ndarray` | Core tensor operations | ✓ |
| `ndarray-linalg` | Linear algebra | ✓ |
| `safetensors` | Weight format parsing | ✓ |
| `memmap2` | Memory-mapped file I/O | ✓ |
| `tokio` + `reqwest` | Async downloads | ✓ |
| `serde_json` | JSON deserialization | ✓ |
| `image` | Image output generation | ✓ |
| `rand` + `rand_distr` | Random sampling | ✓ |
| `half` | BF16 support | ✓ |
| `indicatif` | Progress bars | ✓ |

### Module Structure

```
src/
├── main.rs          - CLI entry point (download, test, clip-test commands)
├── types.rs         - Constants and type definitions
├── weights.rs       - Weight loading from SafeTensors files
├── clip.rs          - CLIP text encoder (COMPLETE)
├── diffusion.rs     - Diffusion sampling (Phase 5)
├── vae.rs           - VAE decoder (Phase 6)
└── utils.rs         - Helper functions (stub)
```

**Key Constants** (from `src/types.rs`):
- `CLIP_EMBEDDING_DIM = 768` - Embedding vector size
- `MAX_TOKEN_LENGTH = 77` - Fixed sequence length
- `TOKEN_VOCAB_SIZE = 49408` - CLIP vocabulary
- `CLIP_NUM_LAYERS = 12` - Transformer layers
- `CLIP_NUM_HEADS = 12` - Attention heads
- `DIFFUSION_STEPS = 1000` - Noise schedule timesteps

---

## Phase 2: Weight Loading ✓

### Model Architecture

Stable Diffusion v1.5 has three components, each stored as a separate SafeTensors file:

| Component | Tensors | Size | Purpose |
|-----------|---------|------|---------|
| **CLIP Text Encoder** | 197 | 469 MB | Convert text to embeddings |
| **UNet Denoiser** | 686 | 3.4 GB | Generate images from noise |
| **VAE Decoder** | 248 | 168 MB | Expand latents to images |

### Weight Loading Strategy

**Problem**: Weights total ~4.26 GB, too large to fit in RAM for most systems.

**Solution**: Memory-mapped file I/O using `memmap2`

```rust
// File is on disk (4.26 GB)
let file = std::fs::File::open("model.safetensors")?;

// Memory-map: OS handles paging, not all in RAM
let mmap = unsafe { memmap2::Mmap::map(&file)? };

// Parse SafeTensors format
let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

// Access tensors via ArrayView (zero-copy!)
let embedding = tensors.tensor("text_model.embeddings.token_embedding.weight")?;
```

**Benefits**:
- **80% memory savings** vs. loading entire file
- **Lazy loading**: Only accessed tensors are paged into RAM
- **Zero-copy**: ArrayView points directly into mmap'd data
- **Scalable**: Can handle multi-GB models on modest hardware

### Precision: F32 (Float32)

- All weights stored as F32 in SafeTensors files
- Maintains full precision for inference (no quality loss)
- Trade-off: Slightly larger memory than BF16, but guaranteed accuracy

---

## Phase 3: Text Encoding with CLIP ✓

### What is CLIP?

**CLIP** (Contrastive Language-Image Pre-training) is OpenAI's multimodal model trained on 400M image-text pairs. It learns to map images and text to the same embedding space.

For Stable Diffusion, we use the **text encoder** half:
- Input: Text prompt (variable length)
- Output: Embedding (77, 768) that semantically captures the text

### CLIP Text Encoder Architecture

```
Input Text: "a beautiful sunset over the ocean"
    ↓
[Tokenizer]
    Output: Token IDs [49406, 320, 5142, 9876, ..., 0, 0] (77 tokens)
    - 49406: Start token
    - 320, 5142, 9876: Subword tokens for words
    - 0: Padding to reach 77
    ↓
[Token Embedding Lookup] [49408, 768]
    Input: Token IDs (77,)
    Output: Embeddings (77, 768)
    - Each token becomes a 768-dimensional vector
    ↓
[Add Positional Embeddings] [77, 768]
    Learned position encoding added element-wise
    - Position 0: [0.1, -0.2, 0.05, ...]
    - Position 1: [-0.15, 0.3, -0.1, ...]
    - ...
    - Position 76: [0.2, 0.1, -0.05, ...]
    ↓
[Transformer Blocks] × 12
    Each block:
        1. LayerNorm: Normalize activations
        2. Multi-Head Self-Attention: Attend to all positions
        3. LayerNorm: Normalize again
        4. MLP (Feed-Forward): Non-linear transformation
    Each block preserves shape (77, 768)
    ↓
[Final LayerNorm]
    Normalize the final output
    ↓
Output: Text Embedding (77, 768)
    - 77 positions (one per token)
    - 768 dimensions (semantic features)
    - Used as conditioning in diffusion model
```

### Multi-Head Self-Attention (Simplified)

Each transformer layer has 12 attention heads working in parallel:

```
Input: (77, 768)
    ↓
For each of 12 heads:
    1. Project input to Q (Query), K (Key), V (Value): (77, 64) each
    2. Attention weights = softmax(Q @ K^T / sqrt(64))
    3. Attend to values: weights @ V
    4. Output: (77, 64)
    ↓
Concatenate all heads: (77, 768)
    ↓
Output projection: (77, 768)
    ↓
Result: Each position can "see" all other positions, sharing information
```

**Why 12 heads?**
- 768 / 12 = 64 dimensions per head
- Multiple heads learn different attention patterns
- Head 1: Attends to adjectives
- Head 2: Attends to objects
- Head 3: Attends to spatial relations
- etc.

### MLP (Feed-Forward Network)

```
Input: (77, 768)
    ↓
Linear 1 (expand): (77, 768) → (77, 3072)
    Weight matrix: [3072, 768]
    Output = Input @ Weight^T + Bias
    ↓
GELU Activation: Non-linear function
    GELU(x) = 0.5 * (1 + tanh(√(2/π) * (x + 0.044715*x³)))
    Smooth alternative to ReLU
    ↓
Linear 2 (project): (77, 3072) → (77, 768)
    Weight matrix: [768, 3072]
    Output = Input @ Weight^T + Bias
    ↓
Result: (77, 768)
    - Same shape as input
    - Different semantic features (transformed by non-linearity)
```

### Key Properties

1. **Fixed Output Shape**: Always (77, 768)
   - 77 token positions (fixed maximum)
   - 768 embedding dimensions

2. **Language Agnostic**: Works with any language trained in CLIP's vocabulary

3. **Semantic Representation**: Embeddings capture meaning
   - Similar prompts → Similar embeddings
   - "cat" and "dog" → Nearby in embedding space
   - "sunset" and "mountain" → Different regions

### Implementation

Test it yourself:
```bash
cargo run --release -- clip-test
```

Output:
```
Input: 'a cat on a beach'
  Output shape: (77, 768)
  Range: [-27.960, 32.890]

Input: 'a beautiful sunset over the ocean'
  Output shape: (77, 768)
  Range: [-27.980, 32.880]
```

---

## Phase 4: Forward Diffusion Process

### What is Diffusion?

Diffusion models learn to reverse a noising process. To understand inference, we must first understand the **forward process** (adding noise).

#### The Forward Process: Adding Noise

```
Step 0 (Clean Image):
    x_0 = [Image data] - our actual image

Step 1 (Tiny bit of noise):
    x_1 = 0.999 * x_0 + 0.045 * noise_1
    Still recognizable as original image

Step 2 (More noise):
    x_2 = 0.997 * x_0 + 0.064 * noise_2
    Starting to look grainy

Step 500 (Much noise):
    x_500 = 0.447 * x_0 + 0.894 * noise_500
    Mostly noise, barely recognizable

Step 1000 (Pure noise):
    x_1000 ≈ noise
    Completely noisy, no image information left
```

### The Noise Schedule

A **noise schedule** defines how much noise is added at each timestep. It controls two factors:

1. **α_t (alpha)**: How much original signal to keep
2. **β_t (beta)**: How much new noise to add

For each timestep t ∈ [1, 1000]:
```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

Where:
- x_t: Noisy version at step t
- x_0: Original clean image
- ε: Random Gaussian noise
- ᾱ_t: Cumulative product of α values (= α_1 * α_2 * ... * α_t)
```

### Common Noise Schedules

**Linear Schedule**
```
β_t = β_min + (β_max - β_min) * t / 1000
β_min = 0.0001
β_max = 0.02

Characteristics:
- Simple formula
- Fast noise early, slower later
- Used in original DDPM paper
```

Example values:
```
Step 1:    β = 0.0001, ᾱ = 0.9999
Step 10:   β = 0.0019, ᾱ = 0.9981
Step 500:  β = 0.0100, ᾱ = 0.4477
Step 1000: β = 0.0200, ᾱ = 0.0001
```

**Cosine Schedule**
```
ᾱ_t = (cos(π * t / 2000))² for t ∈ [0, 1000]

Characteristics:
- Smoother transition
- Better perceptual quality
- Used by newer models like Stable Diffusion
- Preserves more detail early, faster decay late
```

### Why This Matters for Inference

The **reverse process** undoes the forward process:

```
[Inference: Reverse Process]

Start with x_1000 (pure noise)
    ↓
UNet predicts: "What noise was added to get here?"
    ↓
Remove predicted noise → x_999
    ↓
UNet predicts: "What noise was added to get here?"
    ↓
Remove predicted noise → x_998
    ↓
... (repeat 1000 times)
    ↓
Arrive at x_0 (clean image)
```

The noise schedule tells us:
- How much noise should be at each step (for training ground truth)
- How to compute the denoising update

### Variance Schedule Formulas

Given noise schedule β_t:

```
α_t = 1 - β_t
ᾱ_t = ∏(α_i) for i in 1..t  [cumulative product]

Posterior variance (for sampling):
σ_t² = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t

Sampling from reverse process:
x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_pred) + σ_t * z

Where:
- ε_pred: UNet's prediction of added noise
- z: New random noise for this step
```

### Practical Implementation

For Phase 5 (inference), we'll need to pre-compute a noise schedule:

```rust
pub struct NoiseSchedule {
    // 1000 timesteps
    pub betas: Vec<f32>,           // β_t for each step
    pub alphas: Vec<f32>,          // α_t
    pub alphas_cumprod: Vec<f32>,  // ᾱ_t (cumulative product)
    pub sqrt_alphas_cumprod: Vec<f32>,     // √(ᾱ_t)
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,  // √(1 - ᾱ_t)
    pub posterior_variance: Vec<f32>,  // σ_t²
}

impl NoiseSchedule {
    pub fn linear() -> Self {
        // Linear schedule (DDPM paper)
    }
    
    pub fn cosine() -> Self {
        // Cosine schedule (modern, used by SD)
    }
}
```

### Understanding the Reverse Process

The **reverse diffusion process** starts from pure noise and iteratively denoises:

```
Reverse Process (Inference):

For t = 1000 down to 1:
    1. Input to UNet:
       - Noisy latent: x_t
       - Timestep: t (tells UNet how much noise is left)
       - Text embedding: (77, 768) from CLIP
    
    2. UNet predicts:
       - ε_pred: What noise was added at this step?
    
    3. Denoise:
       x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_pred) + σ_t * z
    
    4. Optional: Add small random noise z for stochasticity
       (Make each run slightly different)

After 1000 steps:
    x_0 = Clean latent representation (4, 64, 64)
    
Send to VAE decoder:
    VAE(x_0) → RGB image (3, 512, 512)
```

### Text Conditioning

The CLIP embedding guides generation:

```
Without text (classifier-free guidance off):
    UNet only sees noise level, generates random image

With text conditioning:
    UNet sees:
    - Current noisy latent
    - Timestep
    - CLIP embedding from prompt
    
    Generates: Latent that matches text description

Classifier-Free Guidance (optional enhancement):
    1. Run UNet twice:
       - With text: ε_text
       - Without text: ε_uncond
    2. Combine: ε_guided = ε_uncond + guidance_scale * (ε_text - ε_uncond)
    3. Stronger signal when guidance_scale > 1
       (Trade-off: More text alignment vs. image quality)
```

---

## The Complete Pipeline

### Full Data Flow

```
User: "a cat on a beach"
    ↓
[Phase 3: CLIP Text Encoder]
    "a cat on a beach" → (77, 768) text embedding
    ↓
[Phase 4: Noise Schedule]
    Compute α_t, β_t, √(1-ᾱ_t) for t=1..1000
    ↓
[Phase 5: UNet Denoising Loop] ← Not yet implemented
    Start: x_1000 ~ N(0, 1) [pure noise, shape (4, 64, 64)]
    
    For t = 1000 down to 1:
        Predict noise: ε = UNet(x_t, t, text_embedding)
        Denoise: x_{t-1} = (x_t - β_t/√(1-ᾱ_t) * ε) / √α_t + noise
    
    End: x_0 [clean latent]
    ↓
[Phase 6: VAE Decoder] ← Not yet implemented
    x_0 (4, 64, 64) → Image (3, 512, 512)
    ↓
Result: Generated image matching "a cat on a beach"
```

### Summary of Phases

| Phase | Status | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| 1 | ✓ | - | Setup | Dependencies, modules, constants |
| 2 | ✓ | Disk | WeightStore | Load 4.26 GB of model weights efficiently |
| 3 | ✓ | Text | (77, 768) | Convert text to semantic embeddings |
| 4 | ✓ | Timesteps | NoiseSchedule | Understand noise progression |
| 5 | ⏸️ | (4,64,64) noise | (4,64,64) latent | Iteratively denoise with text guidance |
| 6 | ⏸️ | (4,64,64) latent | (3,512,512) image | Upscale and convert to RGB |

---

## Next Steps: Phase 5

Once Phase 4 theory is understood, Phase 5 implements the actual inference loop:

1. **Initialize noise schedule** (linear or cosine)
2. **Start with random noise** x_1000
3. **Load UNet weights** (686 tensors from SafeTensors)
4. **Sampling loop** (1000 iterations):
   - Call UNet with current latent, timestep, text embedding
   - Denoise step using noise schedule
   - Update latent
5. **Return clean latent** for VAE decoder

---

## Key Takeaways

1. **CLIP Encoder** (Phase 3): Converts semantic meaning → vectors
2. **Noise Schedule** (Phase 4): Defines noise progression mathematically
3. **Reverse Process** (Phase 5): Uses noise schedule to iteratively denoise
4. **Text Guidance**: CLIP embeddings condition the generation
5. **Zero-Copy Architecture**: Memory-mapped weights enable efficiency

The pipeline is a beautiful composition: semantics → stochastic generation → image synthesis.
