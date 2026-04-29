//! Type definitions and constants for Stable Diffusion inference

use ndarray::{Array, ArrayView, IxDyn};

/// Weight matrix: reference to memory-mapped data (no copy needed)
/// 
/// Using ArrayView instead of owned Array allows zero-copy access to weights
/// stored in memory-mapped files. The underlying data is owned by the Mmap.
/// Weights are stored as F32 in the safetensors files.
pub type WeightMatrix<'a> = ArrayView<'a, f32, IxDyn>;

/// Computation tensor: owned array for intermediate computations
/// 
/// Intermediate tensors during inference need to be owned since we modify them.
/// Using F32 to match the precision of weights (no conversion loss).
/// Can be optimized to BF16 later for reduced memory usage.
pub type TensorBf16 = Array<f32, IxDyn>;

/// Single-precision tensors for fallback or precision-critical ops
pub type TensorF32 = Array<f32, IxDyn>;

/// CLIP text embedding dimension
pub const CLIP_EMBEDDING_DIM: usize = 768;

/// Maximum text token length for CLIP
pub const MAX_TOKEN_LENGTH: usize = 77;

/// Token embedding vocabulary size (CLIP uses 49408)
pub const TOKEN_VOCAB_SIZE: usize = 49408;

/// Number of transformer layers in CLIP
pub const CLIP_NUM_LAYERS: usize = 12;

/// Number of attention heads in CLIP
pub const CLIP_NUM_HEADS: usize = 12;

/// MLP expansion ratio in CLIP transformer
pub const CLIP_MLP_EXPANSION: usize = 4;  // 768 → 3072 → 768

/// Diffusion timestep count
pub const DIFFUSION_STEPS: usize = 1000;

/// Latent space height and width
pub const LATENT_HEIGHT: usize = 64;
pub const LATENT_WIDTH: usize = 64;

/// Latent channels
pub const LATENT_CHANNELS: usize = 4;

/// Output image height and width (8x upsampling from latent)
pub const OUTPUT_HEIGHT: usize = 512;
pub const OUTPUT_WIDTH: usize = 512;

/// Output image channels (RGB)
pub const OUTPUT_CHANNELS: usize = 3;
