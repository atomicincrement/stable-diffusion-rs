# Stable Diffusion Demo using ndarray

## Goal
Create a Rust implementation of Stable Diffusion that demonstrates text-to-image generation using only ndarray for tensor operations. Focus on forward inference only.

## High-Level Pipeline
1. Text Prompt → CLIP Encoder → Text Embedding
2. Noise + Text Embedding → Diffusion Process (reverse) → Latent Image
3. Latent Image → VAE Decoder → Final Image

---

## Phase 1: Project Setup & Dependencies ✓ COMPLETE

### Dependencies to add to Cargo.toml
- [x] `ndarray` - Tensor operations
- [x] `ndarray-linalg` - Linear algebra (matrix operations)
- [x] `rand` / `rand_distr` - Random sampling for diffusion
- [x] `serde` + `serde_json` - Weight deserialization
- [x] `image` - Output image generation (png/jpg)
- [x] `tokenizers` - Text tokenization for CLIP
- [x] `ndarray-stats` - Statistical operations
- [x] `half` - BF16 (bfloat16) support for reduced precision compute
- [x] `memmap2` - Memory-mapped file I/O
- [x] `safetensors` - SafeTensors format support
- [x] `reqwest` + `tokio` - Async HTTP for downloads
- [x] `indicatif` - Progress bars for downloads

### Create Module Structure
- [x] `src/main.rs` - Entry point with CLI commands
- [x] `src/types.rs` - Type definitions and constants
- [x] `src/weights.rs` - Weight loading infrastructure
- [x] `src/clip.rs` - Text encoder module (stub)
- [x] `src/diffusion.rs` - Diffusion module (stub)
- [x] `src/vae.rs` - VAE decoder module (stub)
- [x] `src/utils.rs` - Utility functions (stub)
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

## Phase 2: Weight Fetching & Loading ⏳ IN PROGRESS

### 2.1 Fetch Pretrained Weights
- [x] Infrastructure for downloading from Hugging Face Hub
- [x] CLI command: `cargo run -- download`
- [x] Error handling with helpful messages
- [x] Progress bars for downloads
- [x] Actually working download (requires HF_TOKEN authentication)
- [x] Documentation in WEIGHTS.md
- [x] Store locally in `./weights/` directory

### 2.2 Weight Decoding (`weights.rs`) ⏳ IN PROGRESS
- [x] **Precision: Use BF16 (bfloat16) as default**
  - [x] BF16 type alias in types.rs
  - [x] Documentation of benefits (50% memory savings, better stability)
- [x] **Memory-Mapped File Loading (Complete)**
  - [x] `memmap2` dependency added
  - [x] Memory mapping implementation in load_from_safetensors()
  - [x] Lazy loading support
  - [x] ~80% memory savings documentation
  - [x] docs/memory_layout.md with detailed explanation
- [x] **ArrayView for Zero-Copy Access**
  - [x] WeightMatrix<'a> = ArrayView<'a, bf16, IxDyn> type alias
  - [x] Documentation of memory layout
  - [x] Example code in examples/mmap_arrayview.rs
  - [x] Safety guarantees documented
- [x] Structure: WeightStore struct created
- [ ] Load actual weights and validate shapes

---

## Phase 3: Text Encoder (CLIP) ⏸️ NOT STARTED

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

## Phase 4: Latent Diffusion Forward Process (Understanding) ⏸️ NOT STARTED

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

## Phase 5: Latent Diffusion Inference (Reverse Process) ⏸️ NOT STARTED

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

## Phase 6: VAE Decoder ⏸️ NOT STARTED

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

## Phase 7: Putting It Together - Demo (`main.rs`) ⏳ PARTIAL

### 7.1 CLI Interface
- [x] Help message
- [x] `cargo run -- download` command structure
- [x] `cargo run -- test` command structure
- [ ] `cargo run -- generate` command (not implemented)
```
cargo run --release -- --prompt "a cat on a beach" --steps 50 --output out.png
```

### 7.2 Main Function Flow
- [ ] Parse CLI arguments: prompt, num_diffusion_steps, seed, output_path
- [ ] Load weights (CLIP, UNet, VAE) from disk
- [ ] Tokenize and encode text with CLIP → embedding (1, 77, 768)
- [ ] Initialize latent noise: random (1, 4, 64, 64)
- [ ] Run diffusion inference loop with CLIP embedding for conditioning
- [ ] Apply VAE decoder to get RGB image
- [ ] Save image as PNG/JPG
- [ ] Print timing information

### 7.3 Performance Considerations
- [x] **Primary: Use BF16 precision** design
- [ ] CLI flag to switch between BF16 and FP32: `--precision bf16|fp32`
- [ ] Seed parameter for reproducibility
- [ ] Diffusion step parameter (10-50 steps)
- [ ] Model size selection (full vs distilled)

---

## Phase 8: Testing & Validation ⏸️ NOT STARTED

### Test Cases
- [ ] **Weight Loading**: Verify weights load correctly with expected shapes
- [ ] **CLIP Encoding**: Test text embedding output and verify reasonableness
- [ ] **Diffusion Sampling**: Verify noise schedule computation
- [ ] **Image Generation**: Generate test image and verify output shape
- [ ] **End-to-End**: Full pipeline produces valid PNG image

### Debugging Tools
- [ ] Print tensor shapes at each stage
- [ ] Visualize intermediate latents (save as images)
- [ ] Compare outputs with known implementations
- [ ] Test with simple prompts first

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
- Use type aliases for clarity: `type Tensor = Array<bf16, IxDyn>;` (or generic over precision)
- Document expected tensor shapes in function signatures

---

## Stretch Goals

### Goal 0: Custom SafeTensors Parser (High Priority) 🎯 DEFINED

**Status**: Framework defined, not started
**Priority**: High (enables future optimization)
**Effort**: Medium (2-3 days)
**Benefits**: Zero-copy tensor construction, full memory control, reduce dependencies

### COMPLETED ITEMS

#### Goal 0: Custom SafeTensors Parser (High Priority)
Implement safetensors format parsing by hand instead of using external crate:
- **Why**: Educational, optimizable, and reduces dependencies
- **Format Analysis**:
  - Header: 8 bytes (little-endian u64) containing header size
  - Metadata: JSON describing tensor names, shapes, dtype, offsets
  - Data: Raw tensor bytes in specified order
  - Simple format makes hand-parsing feasible
- **Implementation**:
  ```rust
  // Parse safetensors format
  fn parse_safetensors(mmap: &[u8]) -> Result<HashMap<String, Tensor>> {
      let header_size = u64::from_le_bytes(mmap[0..8].try_into()?);
      let header_json = std::str::from_utf8(&mmap[8..8+header_size as usize])?;
      let metadata: SafeTensorsHeader = serde_json::from_str(header_json)?;
      // Map tensor data from mmap using offsets
  }
  ```
- **Benefits**:
  1. Full control over memory layout and tensor buffer management
  2. Can optimize for specific access patterns (sequential vs random access)
  3. Reduce dependencies and binary size
  4. Better integration with ndarray (direct buffer wrapping without copies)
  5. Safer: validate format at compile time
- **Testing**: Compare output with official safetensors crate on sample files
- **Future**: Enable zero-copy tensor construction directly from mmap

### Goal 1: AVX-512-BF16 Optimization 🎯 DEFINED

**Status**: Framework defined, not started
**Priority**: Medium (for CPU performance)
**Effort**: Medium (3-4 days)
**Benefits**: 5-10x speedup on capable hardware

### Goal 1: AVX-512-BF16 Optimization (Details)
Optimize matrix operations to use Intel AVX-512 with native BF16 support:
- **Why**: AVX-512-BF16 provides native hardware acceleration for bfloat16 operations
- **Implementation**:
  - Create `unsafe` SIMD blocks targeting `target_feature = "avx512bf16"`
  - Implement hand-optimized matrix multiplication kernels
  - Use `packed_simd` crate or inline assembly for AVX-512 operations
  - Implement cache-friendly tile-based matrix multiplication
- **Expected Benefit**: 5-10x speedup on capable hardware compared to generic ndarray ops
- **Compatibility**: Add runtime CPU feature detection; fallback to generic ndarray on unsupported CPUs
- **Testing**: Provide benchmark suite (`--bench`) comparing AVX-512 vs generic implementations

### Goal 2: WebGPU Support 🎯 DEFINED

**Status**: Framework defined, not started
**Priority**: Medium (for GPU acceleration)
**Effort**: High (5-7 days)
**Benefits**: 10-50x speedup on discrete GPUs, WebAssembly support

### Goal 2: WebGPU Support (Details)
Port compute-heavy operations to WebGPU for cross-platform GPU acceleration:
- **Why**: WebGPU enables deployment on web and provides portable GPU compute
- **Architecture**:
  - Create optional `wgpu` backend module alongside ndarray
  - Implement key kernels in WGSL (WebGPU Shading Language):
    - Matrix multiplication (gemm)
    - Softmax for attention
    - Convolution operations
    - Element-wise operations (GELU, layer norm)
  - Maintain ndarray backend for CPU fallback
- **Implementation Path**:
  1. Start with most expensive ops: UNet forward pass and attention
  2. Implement buffer management and GPU memory pooling
  3. Create abstraction layer to swap between GPU/CPU backends
  4. Use `wgpu` crate for WebGPU + Vulkan/Metal/DX12 support
- **Expected Benefit**: 10-50x speedup on discrete GPUs
- **Deployment**: Can be compiled to WebAssembly and run in browser
- **Testing**: Verify GPU outputs match CPU results (account for precision differences)

### Integration Strategy
- Use feature flags in Cargo.toml:
  ```toml
  [features]
  avx512-bf16 = []          # Enable AVX-512-BF16 optimizations
  wgpu-backend = ["wgpu"]   # Enable WebGPU/GPU compute
  ```
- Compile with: `cargo build --release --features "avx512-bf16"` or `cargo build --target wasm32-unknown-unknown --features "wgpu-backend"`
- Add benchmark: `cargo bench` to measure performance of different backends
