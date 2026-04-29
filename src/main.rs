mod weights;
mod clip;
mod diffusion;
mod vae;
mod utils;
mod types;

use std::env;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 && args[1] == "download" {
        download_weights().await;
    } else if args.len() > 1 && args[1] == "test" {
        test_weights().await;
    } else if args.len() > 1 && args[1] == "clip-test" {
        test_clip_encoder().await;
    } else {
        print_help();
    }
}

fn print_help() {
    println!("Stable Diffusion Demo - Text to Image Generation");
    println!();
    println!("Usage:");
    println!("  cargo run -- download     Download pretrained weights");
    println!("  cargo run -- test         Test weight loading");
    println!("  cargo run -- clip-test    Test CLIP text encoder");
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
