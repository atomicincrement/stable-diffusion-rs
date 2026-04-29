//! Type definitions and constants for Stable Diffusion inference

use ndarray::Array;
use half::bf16;

/// Default tensor precision: bfloat16 for efficiency and stability
pub type TensorBf16 = Array<bf16, ndarray::IxDyn>;

/// Single-precision tensors for fallback or precision-critical ops
pub type TensorF32 = Array<f32, ndarray::IxDyn>;

/// CLIP text embedding dimension
pub const CLIP_EMBEDDING_DIM: usize = 768;

/// Maximum text token length for CLIP
pub const MAX_TOKEN_LENGTH: usize = 77;

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
