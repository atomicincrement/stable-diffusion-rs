//! Diffusion process for latent image generation

use crate::types::LATENT_CHANNELS;
use ndarray::{Array1, Array2, Array4, Array3};
use std::f32::consts::PI;
use std::collections::HashMap;

/// Noise schedule for diffusion process (1000 timesteps)
pub struct NoiseSchedule {
    /// β_t: Variance added at each step
    pub betas: Array1<f32>,
    /// α_t = 1 - β_t: Signal retention at each step
    pub alphas: Array1<f32>,
    /// ᾱ_t: Cumulative product of alphas (alpha_bar)
    pub alphas_cumprod: Array1<f32>,
    /// √(ᾱ_t): Square root for noise level calculation
    pub sqrt_alphas_cumprod: Array1<f32>,
    /// √(1 - ᾱ_t): Square root of noise coefficient
    pub sqrt_one_minus_alphas_cumprod: Array1<f32>,
    /// Posterior variance σ_t²: variance for reverse process sampling
    pub posterior_variance: Array1<f32>,
}

impl NoiseSchedule {
    /// Create a linear noise schedule (β_t linear from 0.0001 to 0.02)
    /// Used in original DDPM paper
    pub fn linear(num_steps: usize) -> Self {
        let beta_start = 0.0001f32;
        let beta_end = 0.02f32;

        let mut betas = Array1::zeros(num_steps);
        for i in 0..num_steps {
            betas[i] = beta_start + (beta_end - beta_start) * (i as f32) / (num_steps - 1) as f32;
        }

        Self::from_betas(betas)
    }

    /// Create a cosine noise schedule
    /// Smoother transitions, better perceptual quality
    /// Used by Stable Diffusion
    pub fn cosine(num_steps: usize) -> Self {
        let s = 0.008f32;
        let mut betas = Array1::zeros(num_steps);

        for i in 0..num_steps {
            let t = (i as f32) / (num_steps as f32);
            let alpha_bar = ((PI / 2.0 * t).cos()).powi(2);
            let alpha_bar_prev = if i == 0 {
                1.0
            } else {
                let t_prev = ((i - 1) as f32) / (num_steps as f32);
                ((PI / 2.0 * t_prev).cos()).powi(2)
            };
            let beta = 1.0 - (alpha_bar / (alpha_bar_prev + s) * (1.0 - s)).min(0.999f32);
            betas[i] = beta.max(0.0001f32); // Clamp to avoid very small values
        }

        Self::from_betas(betas)
    }

    /// Build noise schedule from beta values
    fn from_betas(betas: Array1<f32>) -> Self {
        let num_steps = betas.len();
        let alphas = 1.0 - &betas;
        
        // Compute cumulative product: ᾱ_t = ∏(α_i) for i=1..t
        let mut alphas_cumprod = Array1::zeros(num_steps);
        alphas_cumprod[0] = alphas[0];
        for i in 1..num_steps {
            alphas_cumprod[i] = alphas_cumprod[i - 1] * alphas[i];
        }

        let sqrt_alphas_cumprod = alphas_cumprod.mapv(f32::sqrt);
        let sqrt_one_minus_alphas_cumprod = (1.0 - &alphas_cumprod).mapv(f32::sqrt);

        // Compute posterior variance: σ_t² = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        let mut posterior_variance = Array1::zeros(num_steps);
        for t in 1..num_steps {
            let numerator = (1.0 - alphas_cumprod[t - 1]) * betas[t];
            let denominator = 1.0 - alphas_cumprod[t];
            posterior_variance[t] = (numerator / denominator).max(0.0);
        }

        NoiseSchedule {
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_variance,
        }
    }

    /// Get noise coefficients for timestep t
    pub fn get_scaled_noise(&self, t: usize) -> (f32, f32) {
        // Returns (sqrt(ᾱ_t), sqrt(1 - ᾱ_t)) for noise formula
        (self.sqrt_alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t])
    }
}

/// Timestep embedding using sinusoidal positional encoding
pub struct TimestepEmbedding {
    /// Pre-computed sinusoidal embeddings for 1000 timesteps
    embeddings: Array2<f32>, // (1000, embedding_dim)
}

impl TimestepEmbedding {
    /// Create timestep embeddings with sinusoidal encoding
    /// Formula: sin(pos / 10000^(2i/d)) for even indices
    ///         cos(pos / 10000^(2i/d)) for odd indices
    pub fn new(embedding_dim: usize, num_timesteps: usize) -> Self {
        let mut embeddings = Array2::zeros((num_timesteps, embedding_dim));
        
        for pos in 0..num_timesteps {
            let pos_f = pos as f32;
            for i in (0..embedding_dim).step_by(2) {
                let div_term = (10000.0_f32).powf((i as f32) / (embedding_dim as f32));
                embeddings[[pos, i]] = (pos_f / div_term).sin();
                
                if i + 1 < embedding_dim {
                    embeddings[[pos, i + 1]] = (pos_f / div_term).cos();
                }
            }
        }
        
        TimestepEmbedding { embeddings }
    }

    /// Get embedding for a specific timestep
    pub fn get(&self, timestep: usize) -> Array1<f32> {
        if timestep >= self.embeddings.nrows() {
            // Clamp to last valid timestep
            self.embeddings.row(self.embeddings.nrows() - 1).to_owned()
        } else {
            self.embeddings.row(timestep).to_owned()
        }
    }
}

/// Residual block in UNet
/// Structure: Conv → GroupNorm → SiLU → Conv → Add residual
pub struct ResidualBlock {
    // Weight matrices for two convolution layers
    // In real impl: conv_in, norm, conv_out
    // For simplicity: store as dense matrices
    pub in_channels: usize,
    pub out_channels: usize,
}

impl ResidualBlock {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        ResidualBlock {
            in_channels,
            out_channels,
        }
    }

    /// Forward pass through residual block (stub - would use conv weights)
    pub fn forward(&self, x: &Array4<f32>, _time_emb: &Array1<f32>) -> Array4<f32> {
        // In real implementation:
        // 1. Apply first conv layer
        // 2. Apply group normalization
        // 3. Apply SiLU activation
        // 4. Add time embedding (broadcasted)
        // 5. Apply second conv layer
        // 6. Add residual connection
        
        // For now, return input unchanged (identity)
        x.clone()
    }
}

/// Cross-attention layer for text conditioning
/// Attention over text embeddings to condition on text
pub struct CrossAttentionBlock {
    pub query_dim: usize,        // Dimension of noisy latent features
    pub context_dim: usize,      // Dimension of text embeddings (768)
    pub num_heads: usize,
}

impl CrossAttentionBlock {
    pub fn new(query_dim: usize, context_dim: usize, num_heads: usize) -> Self {
        CrossAttentionBlock {
            query_dim,
            context_dim,
            num_heads,
        }
    }

    /// Forward pass through cross-attention (stub)
    pub fn forward(
        &self,
        _query: &Array2<f32>,        // (spatial_dims, query_dim)
        _context: &Array2<f32>,      // (77, 768) - text embedding
    ) -> Array2<f32> {
        // In real implementation:
        // 1. Project context to key and value
        // 2. Project query
        // 3. Compute attention weights: softmax(Q @ K^T / sqrt(d))
        // 4. Apply to values: weights @ V
        // 5. Project output back
        
        // For now, return zeros with correct shape
        Array2::zeros((_query.nrows(), self.query_dim))
    }
}

/// UNet denoiser for diffusion - main architecture
pub struct UNetDenoiser {
    /// Timestep embedding module
    pub time_embedding: TimestepEmbedding,
    
    /// Input convolution (4 channels → hidden channels)
    pub input_channels: usize,
    pub hidden_channels: usize,
    
    /// Residual blocks
    pub residual_blocks: Vec<ResidualBlock>,
    
    /// Cross-attention blocks for text conditioning
    pub attention_blocks: Vec<CrossAttentionBlock>,
    
    /// Output convolution (hidden channels → 4 channels)
    pub output_channels: usize,
    
    /// Cached weights from file
    pub weights: HashMap<String, ndarray::Array<f32, ndarray::IxDyn>>,
}

impl UNetDenoiser {
    /// Create a new UNet denoiser with standard Stable Diffusion architecture
    pub fn new() -> Self {
        let input_channels = 4;  // LATENT_CHANNELS
        let hidden_channels = 320; // Standard base width
        let num_residual_blocks = 4;
        let num_attention_blocks = 3;
        
        // Create timestep embedding (embedding_dim = 1280 standard)
        let time_embedding = TimestepEmbedding::new(1280, 1000);
        
        // Create residual blocks
        let mut residual_blocks = Vec::new();
        let mut current_channels = input_channels;
        for _ in 0..num_residual_blocks {
            residual_blocks.push(ResidualBlock::new(current_channels, hidden_channels));
            current_channels = hidden_channels;
        }
        
        // Create attention blocks
        let mut attention_blocks = Vec::new();
        for _ in 0..num_attention_blocks {
            attention_blocks.push(CrossAttentionBlock::new(
                hidden_channels,
                768,  // CLIP_EMBEDDING_DIM
                8,    // Number of attention heads
            ));
        }
        
        UNetDenoiser {
            time_embedding,
            input_channels,
            hidden_channels,
            residual_blocks,
            attention_blocks,
            output_channels: input_channels,
            weights: HashMap::new(),
        }
    }

    /// Load UNet from weights file (safetensors format)
    /// 
    /// Expected tensors include:
    /// - Time embedding layers (128+ tensors)
    /// - Residual block weights (400+ tensors)
    /// - Attention layer weights (150+ tensors)
    /// - Output projection (8+ tensors)
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let mut unet = Self::new();
        
        // Try to load weights from safetensors file
        match std::fs::read(path) {
            Ok(buffer) => {
                // Parse safetensors header to get tensor names
                // In production: deserialize with safetensors crate
                println!("Loading UNet weights from: {}", path);
                
                // Validate file size (should be ~3.4 GB)
                let file_size = buffer.len() as f64 / 1e9;
                if file_size < 3.0 {
                    return Err(format!(
                        "UNet weights file too small: {:.2} GB (expected ~3.4 GB)",
                        file_size
                    ));
                }
                
                println!("✓ UNet weights file loaded: {:.2} GB", file_size);
                
                // TODO: Parse SafeTensors format and load 686 tensors
                // For now, we have the structure but weights are mocked
                Ok(unet)
            }
            Err(e) => {
                println!("Warning: Could not load weights file: {}", e);
                println!("Continuing with random initialization (will produce garbage)");
                Ok(unet)
            }
        }
    }

    /// Predict noise given noisy latent, timestep, and text conditioning
    /// 
    /// # Arguments
    /// * `noisy_latent` - Current noisy latent (1, 4, 64, 64)
    /// * `timestep` - Diffusion timestep (0-999)
    /// * `text_embedding` - Text conditioning (1, 77, 768) from CLIP
    ///
    /// # Returns
    /// Predicted noise with same shape as input latent
    pub fn predict_noise(
        &self,
        noisy_latent: &Array4<f32>,
        timestep: usize,
        text_embedding: &Array2<f32>,
    ) -> Result<Array4<f32>, String> {
        // Validate inputs
        if noisy_latent.dim() != (1, 4, 64, 64) {
            return Err(format!(
                "Invalid latent shape: {:?}, expected (1, 4, 64, 64)",
                noisy_latent.dim()
            ));
        }
        
        if text_embedding.dim() != (77, 768) {
            return Err(format!(
                "Invalid text embedding shape: {:?}, expected (77, 768)",
                text_embedding.dim()
            ));
        }

        // Step 1: Get timestep embedding
        let _time_emb = self.time_embedding.get(timestep);

        // Step 2: Keep latent as (1, 4, 64, 64) for residual blocks
        let mut latent_features = noisy_latent.clone();

        // Step 3: Process through residual blocks with time conditioning
        // NOTE: These are stubs that don't change features yet
        for _res_block in &self.residual_blocks {
            // Stub: in real implementation, would apply conv + norm + time conditioning
            // For now: latent_features = res_block.forward(&latent_features, &_time_emb)
        }

        // Step 4: Flatten for cross-attention processing
        // Shape: (1, 4, 64, 64) → (1, 4096, 4) or process as-is
        let batch_size = latent_features.dim().0;
        let channels = latent_features.dim().1;
        let spatial_size = 64 * 64;
        
        // For attention: flatten spatial but keep batch and channels separate
        let mut attention_features = latent_features
            .clone()
            .into_shape((batch_size * channels, spatial_size))
            .map_err(|e| format!("Failed to reshape for attention: {}", e))?;

        // Step 5: Apply cross-attention with text conditioning
        for _attn_block in &self.attention_blocks {
            // Stub: in real implementation, would apply attention over text embeddings
            // attention_features = attn_block.forward(&attention_features, text_embedding)
        }

        // Step 6: Reshape back to (1, 4, 64, 64) for output
        let noise_pred = attention_features
            .into_shape((batch_size, channels, 64, 64))
            .map_err(|e| format!("Failed to reshape noise prediction back to 4D: {}", e))?;

        Ok(noise_pred)
    }
}

/// Diffusion pipeline for latent image generation
pub struct DiffusionPipeline {
    /// Noise schedule for 1000 steps
    noise_schedule: NoiseSchedule,
    /// UNet denoiser model
    unet: UNetDenoiser,
}

impl DiffusionPipeline {
    /// Create a new diffusion pipeline with linear noise schedule
    pub fn new(unet_path: &str) -> Result<Self, String> {
        let noise_schedule = NoiseSchedule::linear(1000);
        let unet = UNetDenoiser::load_from_file(unet_path)?;

        Ok(DiffusionPipeline {
            noise_schedule,
            unet,
        })
    }

    /// Create pipeline with cosine noise schedule (better quality)
    pub fn with_cosine_schedule(unet_path: &str) -> Result<Self, String> {
        let noise_schedule = NoiseSchedule::cosine(1000);
        let unet = UNetDenoiser::load_from_file(unet_path)?;

        Ok(DiffusionPipeline {
            noise_schedule,
            unet,
        })
    }

    /// Sample latent from diffusion model
    /// 
    /// # Algorithm (DDPM/DDIM Sampling)
    /// 1. Start with x_1000 ~ N(0, 1) - pure noise
    /// 2. For t = 1000 down to 1:
    ///    a. Predict noise using UNet: ε_pred = UNet(x_t, t, text_embedding)
    ///    b. Denoise: x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_pred) + σ_t * z
    ///    c. Add noise for stochasticity (except final step)
    /// 3. Return x_0 (clean latent for VAE decoder)
    pub fn sample(
        &self,
        initial_noise: Array4<f32>,
        text_embedding: &Array2<f32>,
        num_steps: usize,
    ) -> Result<Array4<f32>, String> {
        if num_steps > 1000 {
            return Err(format!("num_steps {} exceeds maximum 1000", num_steps));
        }

        let mut latent = initial_noise;
        let step_size = 1000 / num_steps;

        // Iterate from t=1000 down to t=0
        for step in 0..num_steps {
            let t = 1000 - (step + 1) * step_size;

            println!("Denoising step {}/{} (t={})", step + 1, num_steps, t);

            // Predict noise with UNet
            let noise_pred = self.unet.predict_noise(&latent, t, text_embedding)?;

            // Denoise using predicted noise
            latent = self.denoise_step(&latent, &noise_pred, t)?;
        }

        Ok(latent)
    }

    /// Single denoising step
    /// x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_pred) + σ_t * z
    fn denoise_step(
        &self,
        x_t: &Array4<f32>,
        noise_pred: &Array4<f32>,
        t: usize,
    ) -> Result<Array4<f32>, String> {
        let alpha_t = self.noise_schedule.alphas[t];
        let alpha_t_sqrt = alpha_t.sqrt();
        let beta_t = self.noise_schedule.betas[t];
        let sqrt_one_minus_alpha_bar = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t];

        // Coefficient for noise prediction
        let noise_coeff = beta_t / sqrt_one_minus_alpha_bar;

        // Denoised = (x_t - noise_coeff * noise_pred) / √α_t
        let mut x_denoised = x_t - &(noise_pred * noise_coeff);
        x_denoised = x_denoised / alpha_t_sqrt;

        // For stochasticity, optionally add small noise (except final step)
        // For deterministic generation, skip noise addition
        // For now, we'll add it for variety:
        if t > 1 {
            // Add Gaussian noise scaled by posterior variance
            let variance = self.noise_schedule.posterior_variance[t];
            if variance > 0.0 {
                let noise_std = variance.sqrt();
                // Generate random noise
                let random_noise = generate_noise(x_denoised.dim(), noise_std);
                x_denoised = x_denoised + &random_noise;
            }
        }

        Ok(x_denoised)
    }
}

/// Generate Gaussian noise with given shape and standard deviation
fn generate_noise(shape: (usize, usize, usize, usize), std: f32) -> Array4<f32> {
    use rand_distr::Normal;
    use rand::thread_rng;
    use rand::distributions::Distribution;

    let dist = Normal::new(0.0, std).unwrap();
    let mut rng = thread_rng();

    let mut noise = Array4::zeros(shape);
    for elem in noise.iter_mut() {
        *elem = dist.sample(&mut rng);
    }

    noise
}
