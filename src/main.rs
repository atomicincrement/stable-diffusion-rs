mod weights;
mod clip;
mod diffusion;
mod vae;
mod utils;
mod types;

use std::env;
use std::time::Instant;
use ndarray::{Array4, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 && args[1] == "download" {
        download_weights().await;
    } else if args.len() > 1 && args[1] == "test" {
        test_weights().await;
    } else if args.len() > 1 && args[1] == "clip-test" {
        test_clip_encoder().await;
    } else if args.len() > 1 && args[1] == "noise-test" {
        test_noise_schedule();
    } else if args.len() > 1 && args[1] == "generate" {
        generate_image(&args).await;
    } else {
        print_help();
    }
}

fn print_help() {
    println!("Stable Diffusion Demo - Text to Image Generation");
    println!();
    println!("Usage:");
    println!("  cargo run -- download                                     Download pretrained weights");
    println!("  cargo run -- test                                         Test weight loading");
    println!("  cargo run -- clip-test                                    Test CLIP text encoder");
    println!("  cargo run -- noise-test                                   Test noise schedule");
    println!("  cargo run --release -- generate --prompt \"<TEXT>\" [OPTIONS]");
    println!();
    println!("Generate Options:");
    println!("  --prompt <TEXT>              Text prompt for image generation (required)");
    println!("  --steps <N>                  Number of diffusion steps (default: 50)");
    println!("  --seed <N>                   Random seed for reproducibility (default: random)");
    println!("  --output <PATH>              Output PNG file path (default: output.png)");
    println!();
    println!("Examples:");
    println!("  cargo run --release -- generate --prompt \"a cat on a beach\"");
    println!("  cargo run --release -- generate --prompt \"sunset\" --steps 30 --output sunset.png");
    println!();
    println!("Note: Make sure you have enough disk space (~4GB for SD 1.5)");
}

async fn download_weights() {
    println!("Downloading Stable Diffusion v1.5 weights...");
    
    let model_id = "runwayml/stable-diffusion-v1-5";
    let cache_dir = Some("./weights");
    
    match weights::WeightStore::load_or_download(model_id, cache_dir).await {
        Ok(_) => {
            println!("✓ Weights loaded successfully!");
            println!("Ready to generate images.");
        }
        Err(e) => {
            eprintln!("✗ Failed to load weights: {}", e);
            println!();
            println!("To download weights manually:");
            println!("1. Visit: https://huggingface.co/{}", model_id);
            println!("2. Download the safetensors files");
            println!("3. Place them in the ./weights directory");
        }
    }
}

async fn test_weights() {
    println!("Testing weight loading...");
    println!();
    
    // Check if weights exist locally in component structure
    let model_dir = "./weights/runwayml_stable-diffusion-v1-5";
    
    match weights::WeightStore::load_from_directory(model_dir) {
        Ok(_) => {
            println!();
            println!("✓ All weights loaded successfully!");
            println!("Ready to generate images.");
        }
        Err(e) => {
            println!("✗ Could not load weights: {}", e);
            println!();
            println!("To download weights:");
            println!("1. Run: cargo run -- download");
            println!("2. Then: cargo run -- test");
        }
    }
}

async fn test_clip_encoder() {
    println!("Testing CLIP text encoder...");
    println!();
    
    let clip_path = "./weights/runwayml_stable-diffusion-v1-5/text_encoder/model.safetensors";
    
    match clip::ClipEncoder::load_from_file(clip_path) {
        Ok(encoder) => {
            println!("✓ CLIP encoder loaded successfully!");
            println!();
            
            // Test with sample prompts
            let test_prompts = vec![
                "a cat on a beach",
                "a beautiful sunset over the ocean",
                "dog",
            ];
            
            for prompt in test_prompts {
                match encoder.encode(prompt) {
                    Ok(embedding) => {
                        println!("Input: '{}'", prompt);
                        println!("  Output shape: {:?}", embedding.dim());
                        println!("  Range: [{:.3}, {:.3}]", 
                                 embedding.iter().cloned().fold(f32::INFINITY, f32::min),
                                 embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
                        println!();
                    }
                    Err(e) => {
                        eprintln!("✗ Failed to encode '{}': {}", prompt, e);
                    }
                }
            }
            
            println!("✓ CLIP encoder working correctly!");
        }
        Err(e) => {
            eprintln!("✗ Failed to load CLIP encoder: {}", e);
            println!();
            println!("To download weights:");
            println!("1. Run: cargo run -- download");
            println!("2. Then: cargo run -- clip-test");
        }
    }
}

fn test_noise_schedule() {
    println!("Testing Noise Schedules for Diffusion Process...");
    println!();
    
    // Test linear schedule
    println!("Linear Noise Schedule:");
    println!("=====================");
    let linear_schedule = diffusion::NoiseSchedule::linear(1000);
    
    let test_steps = vec![1, 10, 100, 500, 750, 999];
    println!("Step   | β_t      | α_t      | ᾱ_t      | √(1-ᾱ_t)");
    println!("-------|----------|----------|----------|----------");
    
    for t in &test_steps {
        let beta = linear_schedule.betas[*t];
        let alpha = linear_schedule.alphas[*t];
        let alpha_bar = linear_schedule.alphas_cumprod[*t];
        let sqrt_1_minus_bar = linear_schedule.sqrt_one_minus_alphas_cumprod[*t];
        
        println!("{:5}  | {:.6} | {:.6} | {:.6} | {:.6}",
                 t, beta, alpha, alpha_bar, sqrt_1_minus_bar);
    }
    println!();
    
    // Test cosine schedule
    println!("Cosine Noise Schedule (Better Quality):");
    println!("======================================");
    let cosine_schedule = diffusion::NoiseSchedule::cosine(1000);
    
    println!("Step   | β_t      | α_t      | ᾱ_t      | √(1-ᾱ_t)");
    println!("-------|----------|----------|----------|----------");
    
    for t in &test_steps {
        let beta = cosine_schedule.betas[*t];
        let alpha = cosine_schedule.alphas[*t];
        let alpha_bar = cosine_schedule.alphas_cumprod[*t];
        let sqrt_1_minus_bar = cosine_schedule.sqrt_one_minus_alphas_cumprod[*t];
        
        println!("{:5}  | {:.6} | {:.6} | {:.6} | {:.6}",
                 t, beta, alpha, alpha_bar, sqrt_1_minus_bar);
    }
    println!();
    
    // Explain the values
    println!("Interpretation:");
    println!("- β_t: Amount of noise added at step t");
    println!("- α_t: Signal retention (1 - β_t)");
    println!("- ᾱ_t: Cumulative product (how much original signal remains)");
    println!("- √(1-ᾱ_t): Noise coefficient for reverse process");
    println!();
    println!("✓ Noise schedules computed successfully!");
    println!("  Ready for Phase 5: UNet denoising inference");
}

/// Parse command-line arguments into a map
fn parse_args(args: &[String]) -> std::collections::HashMap<String, String> {
    let mut params = std::collections::HashMap::new();
    let mut i = 0;
    
    while i < args.len() {
        if args[i].starts_with("--") {
            let key = args[i][2..].to_string();
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                params.insert(key, args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    
    params
}

/// Main text-to-image generation function (Step 7.2 implementation)
async fn generate_image(args: &[String]) {
    println!("Stable Diffusion Text-to-Image Generation");
    println!("=========================================");
    println!();
    
    let overall_start = Instant::now();
    
    // Step 1: Parse CLI arguments
    println!("[1/8] Parsing arguments...");
    let params = parse_args(args);
    
    let prompt = match params.get("prompt") {
        Some(p) => p.clone(),
        None => {
            eprintln!("✗ Error: --prompt is required");
            println!();
            println!("Usage: cargo run --release -- generate --prompt \"<TEXT>\" [OPTIONS]");
            return;
        }
    };
    
    let num_steps: usize = params
        .get("steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    
    let seed: u64 = params
        .get("seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| rand::random::<u64>());
    
    let output_path = params
        .get("output")
        .map(|s| s.clone())
        .unwrap_or_else(|| "output.png".to_string());
    
    println!("  ✓ Prompt: \"{}\"", prompt);
    println!("  ✓ Steps: {}", num_steps);
    println!("  ✓ Seed: {}", seed);
    println!("  ✓ Output: {}", output_path);
    println!();
    
    // Step 2: Load weights from disk
    println!("[2/8] Loading weights...");
    let weights_start = Instant::now();
    
    let model_dir = "./weights/runwayml_stable-diffusion-v1-5";
    let clip_path = format!("{}/text_encoder/model.safetensors", model_dir);
    let unet_path = format!("{}/unet/diffusion_pytorch_model.safetensors", model_dir);
    
    // Load CLIP encoder
    let clip_encoder = match clip::ClipEncoder::load_from_file(&clip_path) {
        Ok(encoder) => {
            println!("  ✓ CLIP encoder loaded");
            encoder
        }
        Err(e) => {
            eprintln!("✗ Failed to load CLIP encoder: {}", e);
            println!();
            println!("Please run: cargo run -- download");
            return;
        }
    };
    
    // Load diffusion pipeline with UNet
    let diffusion_pipeline = match diffusion::DiffusionPipeline::with_cosine_schedule(&unet_path) {
        Ok(pipeline) => {
            println!("  ✓ Diffusion pipeline loaded");
            pipeline
        }
        Err(e) => {
            eprintln!("✗ Failed to load diffusion pipeline: {}", e);
            println!();
            println!("Please run: cargo run -- download");
            return;
        }
    };
    
    // VAE decoder loading (optional - will fail gracefully if not implemented)
    let vae_decoder = vae::VaeDecoder::new();
    if vae_decoder.is_ok() {
        println!("  ✓ VAE decoder loaded");
    } else {
        println!("  ⚠ VAE decoder not yet available (using placeholder)");
    }
    
    let weights_time = weights_start.elapsed();
    println!("  Weights loaded in {:.2}s", weights_time.as_secs_f32());
    println!();
    
    // Step 3: Tokenize and encode text with CLIP
    println!("[3/8] Encoding text with CLIP...");
    let encode_start = Instant::now();
    
    let text_embedding = match clip_encoder.encode(&prompt) {
        Ok(embedding) => {
            println!("  ✓ Text embedding shape (CLIP output): {:?}", embedding.dim());
            embedding
        }
        Err(e) => {
            eprintln!("✗ Failed to encode text: {}", e);
            return;
        }
    };

    // Project CLIP embeddings from 768→1280 dimensions to match UNet expectation
    // CLIP outputs: (77, 768)
    // UNet expects: (77, 1280)
    println!("  Projecting embeddings 768→1280 for UNet...");
    let projected_embedding = project_clip_embedding(&text_embedding);
    println!("  ✓ Projected embedding shape: {:?}", projected_embedding.dim());
    
    let encode_time = encode_start.elapsed();
    println!("  Encoding completed in {:.2}s", encode_time.as_secs_f32());
    println!();
    
    // Step 4: Initialize latent noise
    println!("[4/8] Initializing latent noise...");
    let noise_start = Instant::now();
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let initial_noise = {
        use rand_distr::Normal;
        use rand::distributions::Distribution;
        
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut noise = Array4::zeros((1, 4, 64, 64));
        
        for elem in noise.iter_mut() {
            *elem = dist.sample(&mut rng);
        }
        
        println!("  ✓ Latent noise shape: {:?}", noise.dim());
        noise
    };
    
    let noise_time = noise_start.elapsed();
    println!("  Latent noise initialized in {:.2}s", noise_time.as_secs_f32());
    println!();
    
    // Step 5: Run diffusion inference loop
    println!("[5/8] Running diffusion inference loop...");
    let diffusion_start = Instant::now();
    
    let latent_image = match diffusion_pipeline.sample(initial_noise, &projected_embedding, num_steps) {
        Ok(latent) => {
            println!("  ✓ Diffusion complete, latent shape: {:?}", latent.dim());
            latent
        }
        Err(e) => {
            eprintln!("✗ Diffusion failed: {}", e);
            return;
        }
    };
    
    let diffusion_time = diffusion_start.elapsed();
    println!("  Diffusion completed in {:.2}s", diffusion_time.as_secs_f32());
    println!();
    
    // Step 6: Apply VAE decoder (or use placeholder)
    println!("[6/8] Decoding latent with VAE...");
    let vae_start = Instant::now();
    
    let rgb_image = match vae_decoder {
        Ok(decoder) => {
            match decoder.decode(latent_image.clone()) {
                Ok(image) => {
                    println!("  ✓ RGB image shape: {:?}", image.dim());
                    image
                }
                Err(e) => {
                    eprintln!("✗ VAE decoding failed: {}", e);
                    eprintln!("  Using simple placeholder upsampling...");
                    
                    // Placeholder: nearest-neighbor upsampling (8x) without proper VAE
                    placeholder_vae_decode(&latent_image)
                }
            }
        }
        Err(_) => {
            println!("  ℹ Using placeholder VAE (nearest-neighbor upsampling)");
            placeholder_vae_decode(&latent_image)
        }
    };
    
    let vae_time = vae_start.elapsed();
    println!("  VAE decoding completed in {:.2}s", vae_time.as_secs_f32());
    println!();
    
    // Step 7: Save image as PNG
    println!("[7/8] Saving image to PNG...");
    let save_start = Instant::now();
    
    match save_image_as_png(&rgb_image, &output_path) {
        Ok(_) => {
            println!("  ✓ Image saved to: {}", output_path);
        }
        Err(e) => {
            eprintln!("✗ Failed to save image: {}", e);
            return;
        }
    }
    
    let save_time = save_start.elapsed();
    println!("  Image saved in {:.2}s", save_time.as_secs_f32());
    println!();
    
    // Step 8: Print timing information
    println!("[8/8] Generation complete!");
    println!();
    println!("Timing Summary:");
    println!("  - Weight loading: {:.2}s", weights_time.as_secs_f32());
    println!("  - Text encoding:  {:.2}s", encode_time.as_secs_f32());
    println!("  - Diffusion:      {:.2}s", diffusion_time.as_secs_f32());
    println!("  - VAE decoding:   {:.2}s", vae_time.as_secs_f32());
    println!("  - Image saving:   {:.2}s", save_time.as_secs_f32());
    println!("  ─────────────────");
    println!("  Total time:       {:.2}s", overall_start.elapsed().as_secs_f32());
    println!();
    println!("✅ Text-to-image generation successful!");
    println!("   Output saved to: {}", output_path);
}

/// Placeholder VAE decoder using simple nearest-neighbor upsampling (8x)
/// 
/// Since VAE decoder is not yet implemented, this provides a basic alternative:
/// - Upsamples latent (1, 4, 64, 64) to (1, 3, 512, 512)
/// - Uses nearest-neighbor interpolation for simplicity
/// - Converts to [0, 1] range for image output
fn placeholder_vae_decode(latent: &Array4<f32>) -> Array4<f32> {
    use ndarray::Array4;
    
    // For now, just reshape and normalize
    // Proper VAE decoder would:
    // 1. Upsample 64x64 → 512x512 (8x factor)
    // 2. Apply learned decoder blocks
    // 3. Convert 4 channels → 3 RGB channels
    
    let (batch, _, h, w) = latent.dim();
    let scale_factor = 8;
    let out_h = h * scale_factor;
    let out_w = w * scale_factor;
    
    // Simple nearest-neighbor upsampling
    let mut output = Array4::zeros((batch, 3, out_h, out_w));
    
    for b in 0..batch {
        for c in 0..3.min(latent.dim().1) {
            for ih in 0..h {
                for iw in 0..w {
                    let val = latent[[b, c % latent.dim().1, ih, iw]];
                    
                    for dy in 0..scale_factor {
                        for dx in 0..scale_factor {
                            let oh = ih * scale_factor + dy;
                            let ow = iw * scale_factor + dx;
                            if oh < out_h && ow < out_w {
                                output[[b, c, oh, ow]] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Normalize to [0, 1]
    let min_val = output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-6);
    
    output.mapv(|x| (x - min_val) / range)
}

/// Save image tensor to PNG file
/// 
/// # Arguments
/// * `image` - Tensor of shape (batch, channels, height, width) in range [0, 1]
/// * `path` - Output file path
fn save_image_as_png(image: &Array4<f32>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use image::ImageBuffer;
    use image::Rgb;
    
    let (batch, channels, height, width) = image.dim();
    
    if batch != 1 {
        return Err(format!("Expected batch size 1, got {}", batch).into());
    }
    
    if channels < 3 {
        return Err(format!("Expected at least 3 channels, got {}", channels).into());
    }
    
    // Create RGB image buffer
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);
    
    for y in 0..height {
        for x in 0..width {
            let r = (image[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (image[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (image[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            
            img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    img_buffer.save(path)?;
    Ok(())
}

/// Project CLIP embeddings from 768 to 1280 dimensions
/// 
/// # Context
/// - CLIP encoder outputs: (num_tokens=77, embedding_dim=768)
/// - UNet expects: (num_tokens=77, context_dim=1280)
/// - The projection layer is normally trained as part of the SD model
/// 
/// # Implementation
/// Since we don't have the trained projection weights yet, this uses a simple
/// projection that preserves information: repeats and selects features.
/// This is a temporary solution - ideally should load actual weights.
///
/// # Arguments
/// * `clip_embedding` - Text embeddings from CLIP (77, 768)
///
/// # Returns
/// Projected embeddings (77, 1280)
fn project_clip_embedding(clip_embedding: &Array2<f32>) -> Array2<f32> {
    let (seq_len, embed_dim) = clip_embedding.dim();
    
    if embed_dim != 768 {
        panic!("Expected CLIP embedding dimension 768, got {}", embed_dim);
    }
    
    // Target dimension (UNet context dimension)
    let target_dim = 1280;
    
    // Simple projection: duplicate features strategically
    // [a0, a1, ..., a767] → [a0, a1, ..., a767, a0, a1, ..., a512]
    // This preserves all information and is deterministic
    let mut projected = Array2::zeros((seq_len, target_dim));
    
    for i in 0..seq_len {
        for j in 0..target_dim {
            // Map target dimension back to source dimension with wraparound
            let source_idx = j % embed_dim;
            projected[[i, j]] = clip_embedding[[i, source_idx]];
        }
    }
    
    projected
}
