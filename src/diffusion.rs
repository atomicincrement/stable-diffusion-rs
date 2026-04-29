//! Diffusion process for latent image generation

use crate::types::LATENT_CHANNELS;
use ndarray::{Array1, Array2, Array4};
use std::f32::consts::PI;

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

/// UNet denoiser for diffusion
pub struct UNetDenoiser {
    // TODO: Load UNet weights from safetensors
    // For now, stub implementation
}

impl UNetDenoiser {
    /// Load UNet from weights file
    pub fn load_from_file(_path: &str) -> Result<Self, String> {
        // TODO: Load 686 UNet tensors from safetensors
        Err("UNet loading not yet implemented".to_string())
    }

    /// Predict noise given noisy latent, timestep, and text conditioning
    pub fn predict_noise(
        &self,
        _noisy_latent: &Array4<f32>,
        _timestep: usize,
        _text_embedding: &Array2<f32>,
    ) -> Result<Array4<f32>, String> {
        // TODO: Implement UNet forward pass
        Err("UNet inference not yet implemented".to_string())
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
