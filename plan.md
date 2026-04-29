# Stable Diffusion Demo using ndarray

## Goal
Create a Rust implementation of Stable Diffusion that demonstrates text-to-image generation using only ndarray for tensor operations. Focus on forward inference only.

## High-Level Pipeline
1. Text Prompt → CLIP Encoder → Text Embedding
2. Noise + Text Embedding → Diffusion Process (reverse) → Latent Image
3. Latent Image → VAE Decoder → Final Image

---

## Phase 1: Project Setup & Dependencies

### Dependencies to add to Cargo.toml
- `ndarray` - Tensor operations
- `ndarray-linalg` - Linear algebra (matrix operations)
- `rand` / `rand_distr` - Random sampling for diffusion
- `serde` + `serde_json` - Weight deserialization
- `image` - Output image generation (png/jpg)
- `tokenizers` - Text tokenization for CLIP
- `ndarray-stats` - Statistical operations

### Create Module Structure
```
src/
├── main.rs          - Entry point, CLI demo
├── weights.rs       - Weight loading and parsing
├── clip.rs          - Text encoder (CLIP model)
├── diffusion.rs     - Diffusion process and inference
├── vae.rs           - VAE decoder
├── utils.rs         - Common utilities (tensor operations, normalization)
└── types.rs         - Type definitions and constants
```

---

## Phase 2: Weight Fetching & Loading

### 2.1 Fetch Pretrained Weights
- Find a small Stable Diffusion checkpoint (e.g., SD 1.5 with pruned weights or distilled version)
- Options:
  - Hugging Face diffusers (e.g., `runwayml/stable-diffusion-v1-5`)
  - Direct checkpoint download (safetensors or .pt format)
- Store locally in `./weights/` directory
- Document source and licensing

### 2.2 Weight Decoding (`weights.rs`)
- Parse weight files (safetensors format is preferred - easier to parse than PyTorch)
- Deserialize into ndarray tensors
- Structure: Create a `WeightStore` struct containing:
  - CLIP text encoder weights
  - Diffusion UNet weights
  - VAE decoder weights
- Load into memory and validate shapes
- Consider: How to handle different precision types (fp32, fp16)

---

## Phase 3: Text Encoder (CLIP)

### 3.1 CLIP Architecture Overview
- Input: Tokenized text (max 77 tokens)
- Output: 768-dim text embedding (conditioning vector)
- Components:
  - Token embedding layer
  - Positional encoding
  - Transformer encoder (12 layers)
  - Final layer norm + projection

### 3.2 Implementation (`clip.rs`)
- Tokenizer: Use `tokenizers` crate to tokenize input text
  - Pad/truncate to 77 tokens
  - Handle special tokens (start, end)
- Embedding layer: Look up token embeddings from weight matrix
- Positional encoding: Add learned position embeddings
- Transformer blocks:
  - Multi-head self-attention
  - Feed-forward layers (MLPs)
  - Layer normalization
- Output projection: Project to final embedding dimension
- Return: Shape (77, 768) for text conditioning

### 3.3 Key Implementation Details
- Use matrix multiplication (`ndarray::linalg::general_mat_mul`) for linear layers
- Implement softmax for attention weights
- GELU activation function
- Layer norm: `(x - mean) / sqrt(variance + eps) * gamma + beta`

---

## Phase 4: Latent Diffusion Forward Process (Understanding)

### 4.1 Concepts (Reference, not coded)
- Forward process: Progressively add Gaussian noise to images
- Noise schedule: Beta values at each timestep (1000 steps typically)
- Alpha cumulative product: Used to compute noise level at any step
- Formula: `x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps`

### 4.2 Why This Matters
- Understanding the forward process helps understand reverse inference
- Noise schedule determines denoising progression
- Document: Linear vs cosine vs other schedules

---

## Phase 5: Latent Diffusion Inference (Reverse Process)

### 5.1 Inference Pipeline (`diffusion.rs`)

#### Components needed:
1. **Noise Schedule**: Pre-compute alpha, beta, and alpha_cumprod for all 1000 timesteps
2. **UNet Denoiser**: Takes (noisy_latent, timestep, text_embedding) → prediction
3. **Sampling Loop**: Iteratively denoise using predictions

#### UNet Architecture
- Input: Noisy latent (4, 64, 64) + timestep embedding + text conditioning
- Output: Predicted noise (same shape as input)
- Structure:
  - Timestep embedding: Sinusoidal encoding → MLP
  - Multiple residual blocks
  - Attention layers (spatial + cross-attention with text)
  - Skip connections
  - Progressive upsampling

#### Sampling Algorithm (DDPM/DDIM)
```
1. Start with random noise x_T ~ N(0, 1)
2. For t = 1000 down to 1:
   a. Predict noise using UNet: pred_noise = UNet(x_t, t, text_embedding)
   b. Compute denoised image: x_0_pred = (x_t - sqrt(1-alpha_cumprod) * pred_noise) / sqrt(alpha_cumprod)
   c. Sample next step: x_{t-1} = sqrt(alpha_{t-1}) * x_0_pred + sqrt(1 - alpha_{t-1}) * z
   d. If t > 1: add noise z ~ N(0, 1)
3. Output: x_0 (denoised latent in latent space)
```

### 5.2 Implementation Notes
- Timestep embedding: Use sinusoidal positional encoding
- Cross-attention: Attention over text embedding for conditioning
- Attention implementation: Query, Key, Value projections + softmax
- Use classifier-free guidance (optional enhancement): Run inference twice (with and without text) and blend

---

## Phase 6: VAE Decoder

### 6.1 VAE Architecture
- Input: Latent representation (4, 64, 64)
- Output: RGB image (3, 512, 512) - scales up 8x
- Components:
  - Upsampling blocks (nearest neighbor or transpose conv)
  - Residual blocks with convolutions
  - Final projection to 3 channels
  - Output activation: tanh or sigmoid, scale to [0, 1]

### 6.2 Implementation (`vae.rs`)
- Implement basic convolution operations using ndarray
- Or use simplified upsampling: nearest-neighbor, bilinear, or simple conv
- Normalization layers: similar to layer norm for image tensors
- Load VAE decoder weights and apply sequentially

### 6.3 Output Processing
- Ensure output is normalized to [0, 1]
- Convert to u8 for image format
- Handle color space (likely already RGB)

---

## Phase 7: Putting It Together - Demo (`main.rs`)

### 7.1 CLI Interface
```
cargo run --release -- --prompt "a cat on a beach" --steps 50 --output out.png
```

### 7.2 Main Function Flow
1. Parse CLI arguments: prompt, num_diffusion_steps, seed, output_path
2. Load weights (CLIP, UNet, VAE) from disk
3. Tokenize and encode text with CLIP → embedding (1, 77, 768)
4. Initialize latent noise: random (1, 4, 64, 64)
5. Run diffusion inference loop with CLIP embedding for conditioning
6. Apply VAE decoder to get RGB image
7. Save image as PNG/JPG
8. Print timing information

### 7.3 Performance Considerations
- Consider using seed for reproducibility
- Reduce diffusion steps for faster inference (10-50 steps)
- Use smaller model or distilled version if available
- Consider quantization (fp16) for faster computation

---

## Phase 8: Testing & Validation

### Test Cases
1. **Weight Loading**: Verify all weights load correctly with expected shapes
2. **CLIP Encoding**: Test text embedding output shape and values are reasonable
3. **Diffusion Sampling**: Verify noise schedule is computed correctly
4. **Image Generation**: Generate test image and verify output shape (3, 512, 512)
5. **End-to-End**: Full pipeline produces valid PNG image

### Debugging Tools
- Print tensor shapes at each stage
- Visualize intermediate latents (save as images)
- Compare outputs with known implementations if possible
- Test with simple prompts first

---

## Implementation Tips & Challenges

### Challenges
1. **Matrix Operations**: ndarray doesn't have all deep-learning optimizations (no GPU)
  - Mitigation: Focus on correctness first, optimize later
2. **Convolutions**: May need to implement manually or use simplified version
  - Simpler approach: Use 1x1 convs or fully connected layers
3. **Attention Computation**: Can be memory-intensive
  - Mitigation: Process in chunks or use lower precision
4. **Large Weights File**: May need to manage memory carefully
  - Mitigation: Stream weights if needed, use safetensors format

### Optimization Opportunities
- Multi-threading with `rayon` for parallel operations
- Consider using `ndarray-linalg` for optimized matrix ops
- Profile and identify bottlenecks
- Optional: Create simplified model for faster demo

### Code Organization
- Keep layers and operations pure functions where possible
- Create helper functions for common ops (matrix mul, activation, norm)
- Use type aliases for clarity: `type Tensor = Array<f32, IxDyn>;`
- Document expected tensor shapes in function signatures
