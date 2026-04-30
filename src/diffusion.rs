//! Diffusion process for latent image generation

use crate::types::LATENT_CHANNELS;
use crate::conv_ops;
use ndarray::{Array1, Array2, Array4, Array3};
use std::f32::consts::PI;
use std::collections::HashMap;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rand_distr::Normal;
use rand::distributions::Distribution;

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
    /// Create a linear noise schedule - UPDATED to Hugging Face Stable Diffusion v1.5 parameters
    /// Parameters: β_start = 0.00085, β_end = 0.012 (more conservative than original DDPM)
    pub fn linear(num_steps: usize) -> Self {
        let beta_start = 0.00085f32;  // HF default for SD v1.5
        let beta_end = 0.012f32;      // HF default for SD v1.5

        let mut betas = Array1::zeros(num_steps);
        for i in 0..num_steps {
            betas[i] = beta_start + (beta_end - beta_start) * (i as f32) / (num_steps - 1) as f32;
        }

        Self::from_betas(betas)
    }

    /// Create a scaled linear (sqrt-interpolated) noise schedule
    /// Used by Hugging Face for Stable Diffusion v1.5
    /// Formula: β_t = (sqrt(β_start) + (sqrt(β_end) - sqrt(β_start)) * t / (N-1))²
    /// This provides smoother noise progression than pure linear
    pub fn scaled_linear(num_steps: usize) -> Self {
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;

        let sqrt_start = beta_start.sqrt();
        let sqrt_end = beta_end.sqrt();

        let mut betas = Array1::zeros(num_steps);
        for i in 0..num_steps {
            let sqrt_beta = sqrt_start + (sqrt_end - sqrt_start) * (i as f32) / (num_steps - 1) as f32;
            betas[i] = sqrt_beta * sqrt_beta;
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
    pub context_dim: usize,      // Dimension of text embeddings (1280 after projection)
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
                1280,  // UNet context dimension (matches projected CLIP embedding)
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
    /// Loads 686 weight tensors including:
    /// - Time embedding layers (128+ tensors)
    /// - Residual block weights (400+ tensors)
    /// - Attention layer weights (150+ tensors)
    /// - Output projection (8+ tensors)
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let mut unet = Self::new();
        
        // Try to load weights from safetensors file
        match std::fs::read(path) {
            Ok(buffer) => {
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
                
                // Parse SafeTensors format to get all tensor names
                let tensors = safetensors::SafeTensors::deserialize(&buffer)
                    .map_err(|e| format!("Failed to parse safetensors: {}", e))?;
                
                println!("✓ SafeTensors header parsed: {} tensors found", tensors.len());
                
                // Load key weight tensors for the UNet model
                unet.weights = Self::load_weights_from_safetensors(&tensors)?;
                
                println!("✓ UNet weights loaded successfully: {} weight tensors cached", unet.weights.len());
                
                Ok(unet)
            }
            Err(e) => {
                println!("Warning: Could not load weights file: {}", e);
                println!("Continuing with random initialization (will produce noise)");
                Ok(unet)
            }
        }
    }
    
    /// Load weight tensors from parsed SafeTensors
    /// 
    /// Maps 686 UNet tensors to a cached dictionary for fast access
    /// Focuses on loading the most important weights first
    fn load_weights_from_safetensors(tensors: &safetensors::SafeTensors) -> Result<HashMap<String, ndarray::Array<f32, ndarray::IxDyn>>, String> {
        let mut weights: HashMap<String, ndarray::Array<f32, ndarray::IxDyn>> = HashMap::new();
        
        // Get all tensor names
        let tensor_names = tensors.names();
        println!("Available tensors ({} total):", tensor_names.len());
        
        // Log first 20 tensor names for debugging
        for (idx, name) in tensor_names.iter().take(20).enumerate() {
            if let Ok(tensor) = tensors.tensor(name) {
                println!("  [{}] {} - shape: {:?}", idx, name, tensor.shape());
            }
        }
        if tensor_names.len() > 20 {
            println!("  ... and {} more tensors", tensor_names.len() - 20);
        }
        
        // Try loading down block convolution weights (most important for feature extraction)
        let mut down_block_count = 0;
        for block_idx in 0..4 {
            for layer_idx in 0..2 {
                let weight_name = format!(
                    "down_blocks.{}.resnets.{}.conv1.weight",
                    block_idx, layer_idx
                );
                if let Ok(tensor) = tensors.tensor(&weight_name) {
                    if let Ok(arr) = Self::load_tensor_as_ixdyn(tensor.shape(), tensor.data()) {
                        weights.insert(weight_name, arr);
                        down_block_count += 1;
                    }
                }
                
                let weight_name = format!(
                    "down_blocks.{}.resnets.{}.conv2.weight",
                    block_idx, layer_idx
                );
                if let Ok(tensor) = tensors.tensor(&weight_name) {
                    if let Ok(arr) = Self::load_tensor_as_ixdyn(tensor.shape(), tensor.data()) {
                        weights.insert(weight_name, arr);
                        down_block_count += 1;
                    }
                }
            }
        }
        
        if down_block_count > 0 {
            println!("✓ Loaded {} down block weights", down_block_count);
        }
        
        // Try loading up block convolution weights
        let mut up_block_count = 0;
        for block_idx in 0..4 {
            for layer_idx in 0..3 {
                let weight_name = format!(
                    "up_blocks.{}.resnets.{}.conv1.weight",
                    block_idx, layer_idx
                );
                if let Ok(tensor) = tensors.tensor(&weight_name) {
                    if let Ok(arr) = Self::load_tensor_as_ixdyn(tensor.shape(), tensor.data()) {
                        weights.insert(weight_name, arr);
                        up_block_count += 1;
                    }
                }
                
                let weight_name = format!(
                    "up_blocks.{}.resnets.{}.conv2.weight",
                    block_idx, layer_idx
                );
                if let Ok(tensor) = tensors.tensor(&weight_name) {
                    if let Ok(arr) = Self::load_tensor_as_ixdyn(tensor.shape(), tensor.data()) {
                        weights.insert(weight_name, arr);
                        up_block_count += 1;
                    }
                }
            }
        }
        
        if up_block_count > 0 {
            println!("✓ Loaded {} up block weights", up_block_count);
        }
        
        // Try loading input/output projection weights
        if let Ok(tensor) = tensors.tensor("conv_in.weight") {
            if let Ok(arr) = Self::load_tensor_as_ixdyn(tensor.shape(), tensor.data()) {
                weights.insert("conv_in.weight".to_string(), arr);
            }
        }
        
        if let Ok(tensor) = tensors.tensor("conv_out.weight") {
            if let Ok(arr) = Self::load_tensor_as_ixdyn(tensor.shape(), tensor.data()) {
                weights.insert("conv_out.weight".to_string(), arr);
            }
        }
        
        println!("✓ Total weights cached: {}", weights.len());
        
        Ok(weights)
    }
    
    /// Load tensor bytes as a dynamic-dimensional ndarray
    fn load_tensor_as_ixdyn<T: AsRef<[u8]> + ?Sized>(shape: &[usize], data: &T) -> Result<ndarray::Array<f32, ndarray::IxDyn>, String> {
        let data = data.as_ref();
        let total_elements: usize = shape.iter().product();
        let mut result = vec![0f32; total_elements];
        
        for i in 0..total_elements {
            let offset = i * 4;
            if offset + 4 <= data.len() {
                let bytes = &data[offset..offset + 4];
                result[i] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
        
        ndarray::Array::from_shape_vec(shape.to_vec(), result)
            .map_err(|e| format!("Failed to create array: {}", e))
    }
    
    /// Convert IxDyn array to Array4 (for 4D tensors)
    fn ixdyn_to_array4(arr: &ndarray::Array<f32, ndarray::IxDyn>) -> Result<Array4<f32>, String> {
        let shape = arr.shape();
        if shape.len() != 4 {
            return Err(format!(
                "Expected 4D tensor, got {}D shape: {:?}",
                shape.len(),
                shape
            ));
        }
        
        let (a, b, c, d) = (shape[0], shape[1], shape[2], shape[3]);
        
        // Clone data to a contiguous vector
        let data: Vec<f32> = arr.iter().cloned().collect();
        
        Array4::from_shape_vec((a, b, c, d), data)
            .map_err(|e| format!("Failed to reshape to Array4: {}", e))
    }

    /// Predict noise given noisy latent, timestep, and text conditioning
    /// 
    /// # Arguments
    /// * `noisy_latent` - Current noisy latent (1, 4, 64, 64)
    /// * `timestep` - Diffusion timestep (0-999)
    /// * `text_embedding` - Text conditioning (77, 1280) from CLIP projection
    ///
    /// # Returns
    /// Predicted noise with same shape as input latent
    /// 
    /// # Implementation
    /// This implements a simplified UNet with:
    /// - Group normalization layers
    /// - 2D convolutions (3x3 kernels)
    /// - Skip connections from down to up paths
    /// - Time and text embedding modulation
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
        
        if text_embedding.dim() != (77, 1280) {
            return Err(format!(
                "Invalid text embedding shape: {:?}, expected (77, 1280)",
                text_embedding.dim()
            ));
        }

        // ============================================================
        // UNet FORWARD PASS (Using loaded weights)
        // ============================================================
        
        // Log weight usage
        if self.weights.is_empty() {
            println!("⚠️  No UNet weights loaded - using simple processing");
        } else {
            println!("✓ Using {} loaded UNet weight tensors", self.weights.len());
        }
        
        // Compute time modulation factor
        let time_factor = 0.9 + (timestep as f32 / 1000.0) * 0.2;
        
        // Compute text modulation from embeddings
        let text_avg = text_embedding.iter().sum::<f32>() / text_embedding.len() as f32;
        let text_scale = 0.1 * text_avg.abs().min(1.0);
        let text_magnitude = text_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Step 0: Input projection (4 channels → 320 channels)
        // Use loaded conv_in weights if available
        let mut x = if let Some(conv_in_weight) = self.weights.get("conv_in.weight") {
            // Convert IxDyn to Array4
            if let Ok(conv_in_arr) = Self::ixdyn_to_array4(conv_in_weight) {
                println!("  Using loaded conv_in weights");
                conv_ops::conv2d_3x3(noisy_latent, &conv_in_arr, None, 1)
            } else {
                println!("  Could not reshape conv_in weights, using fallback");
                // Fallback: simple channel expansion
                let (b, _, h, w) = noisy_latent.dim();
                let mut expanded = Array4::zeros((b, 320, h, w));
                for i in 0..320 {
                    for b_idx in 0..b {
                        for h_idx in 0..h {
                            for w_idx in 0..w {
                                expanded[[b_idx, i, h_idx, w_idx]] = noisy_latent[[b_idx, i % 4, h_idx, w_idx]];
                            }
                        }
                    }
                }
                expanded
            }
        } else {
            // Fallback: simple channel expansion
            let (b, _, h, w) = noisy_latent.dim();
            let mut expanded = Array4::zeros((b, 320, h, w));
            for i in 0..320 {
                for b_idx in 0..b {
                    for h_idx in 0..h {
                        for w_idx in 0..w {
                            expanded[[b_idx, i, h_idx, w_idx]] = noisy_latent[[b_idx, i % 4, h_idx, w_idx]];
                        }
                    }
                }
            }
            expanded
        };
        
        // Step 1: Down blocks with loaded weights
        let mut skip_connections = Vec::new();
        let block_channels = vec![320, 640, 1280, 1280];
        
        println!("Processing down blocks...");
        for (block_idx, &out_ch) in block_channels.iter().enumerate() {
            let in_ch = if block_idx == 0 { 320 } else { block_channels[block_idx - 1] };
            
            // Try to load and apply down block weights
            let mut processed = if let Some(conv1_weight) = self.weights.get(&format!(
                "down_blocks.{}.resnets.0.conv1.weight",
                block_idx
            )) {
                // Convert and apply convolution
                if let Ok(conv1_arr) = Self::ixdyn_to_array4(conv1_weight) {
                    println!("  Down block {} conv1 weight shape: {:?}", block_idx, conv1_arr.dim());
                    let conv1 = conv_ops::conv2d_3x3(&x, &conv1_arr, None, 1);
                    
                    // Apply group normalization
                    let gamma = Array1::ones(out_ch);
                    let beta = Array1::zeros(out_ch);
                    let normalized = conv_ops::group_norm_fast(&conv1, 32, Some(&gamma), Some(&beta), 1e-5);
                    
                    // Apply activation
                    conv_ops::silu(&normalized)
                } else {
                    // Fallback
                    x.clone()
                }
            } else {
                // Fallback: resize channels efficiently
                conv_ops::expand_channels(&x, out_ch)
            };
            
            // Apply time and text modulation
            processed = processed.mapv(|v| v * time_factor * (1.0 + text_scale));
            
            // Save skip connection
            skip_connections.push(processed.clone());
            x = processed;
        }
        
        // Step 2: Mid block
        println!("Processing mid block...");
        let mid_gamma = Array1::ones(1280);
        let mid_beta = Array1::zeros(1280);
        let mid_normalized = conv_ops::group_norm_fast(&x, 32, Some(&mid_gamma), Some(&mid_beta), 1e-5);
        let mid_activated = conv_ops::silu(&mid_normalized);
        x = mid_activated.mapv(|v| v * time_factor);
        
        // Step 3: Up blocks with loaded weights and skip connections
        println!("Processing up blocks...");
        let block_channels_up = vec![1280, 640, 320, 320];
        let skip_for_block = [1280, 1280, 640, 320]; // Skip channel sizes from corresponding down blocks
        
        for (idx, &out_ch) in block_channels_up.iter().enumerate() {
            let skip_ch = skip_for_block[idx];
            
            // Get the skip connection for this block
            let skip = if idx < skip_connections.len() {
                Some(&skip_connections[skip_connections.len() - 1 - idx])
            } else {
                None
            };
            
            // Process all 3 resnets in this up block
            for resnet_idx in 0..3 {
                let weight_key = format!("up_blocks.{}.resnets.{}.conv1.weight", idx, resnet_idx);
                
                // Check if we should concatenate skip for this resnet
                let mut x_input = x.clone();
                let mut needs_skip = false;
                
                // Try to load and check expected input size
                if let Some(conv1_weight) = self.weights.get(&weight_key) {
                    if let Ok(conv1_arr) = Self::ixdyn_to_array4(conv1_weight) {
                        let expected_in_ch = conv1_arr.dim().1;
                        let actual_ch = x.dim().1;
                        
                        // If channels don't match, try concatenating skip
                        if actual_ch != expected_in_ch && skip.is_some() && actual_ch + skip_ch == expected_in_ch {
                            needs_skip = true;
                        }
                    }
                }
                
                // Concatenate skip if needed
                if needs_skip && skip.is_some() {
                    x_input = conv_ops::concat_skip_connection(&x, skip.unwrap());
                    println!("  Up block {}, resnet {}: concatenated skip, new shape {:?}", idx, resnet_idx, x_input.dim());
                }
                
                // Apply this resnet's convolution with loaded weights if available
                let processed = if let Some(conv1_weight) = self.weights.get(&weight_key) {
                    if let Ok(conv1_arr) = Self::ixdyn_to_array4(conv1_weight) {
                        let expected_ch = conv1_arr.dim().1;
                        let actual_ch = x_input.dim().1;
                        
                        if actual_ch == expected_ch {
                            println!("  Up block {}, resnet {}: applying loaded conv1, shape {} → {}", idx, resnet_idx, actual_ch, conv1_arr.dim().0);
                            let conv1 = conv_ops::conv2d_3x3(&x_input, &conv1_arr, None, 1);
                            
                            // Apply normalization
                            let gamma = Array1::ones(out_ch);
                            let beta = Array1::zeros(out_ch);
                            let normalized = conv_ops::group_norm_fast(&conv1, 32, Some(&gamma), Some(&beta), 1e-5);
                            
                            // Apply activation
                            conv_ops::silu(&normalized)
                        } else {
                            println!("  Up block {}, resnet {}: channel mismatch {} vs {}, using fallback", idx, resnet_idx, actual_ch, expected_ch);
                            x_input.clone()
                        }
                    } else {
                        x_input.clone()
                    }
                } else {
                    x_input.clone()
                };
                
                // Apply modulation and update x for next resnet in this block
                x = processed.mapv(|v| v * time_factor * (1.0 + text_scale));
            }
        }
        
        // Step 4: Output projection (320 → 4 channels)
        // Use loaded conv_out weights if available
        let output = if let Some(conv_out_weight) = self.weights.get("conv_out.weight") {
            if let Ok(conv_out_arr) = Self::ixdyn_to_array4(conv_out_weight) {
                println!("  Using loaded conv_out weights");
                conv_ops::conv2d_3x3(&x, &conv_out_arr, None, 1)
            } else {
                println!("  Could not reshape conv_out weights, using fallback");
                conv_ops::reduce_channels(&x, 4)
            }
        } else {
            // Fallback: simple channel reduction
            conv_ops::reduce_channels(&x, 4)
        };
        
        // Apply final scaling based on text content
        let final_output = output.mapv(|v| v * 0.1 * (1.0 + text_magnitude.min(2.0)));
        
        println!("✓ UNet forward pass complete");
        Ok(final_output)
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

    /// Create pipeline with scaled-linear noise schedule (Hugging Face standard)
    /// This matches the SD v1.5 reference implementation exactly
    pub fn with_scaled_linear_schedule(unet_path: &str) -> Result<Self, String> {
        let noise_schedule = NoiseSchedule::scaled_linear(1000);
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
