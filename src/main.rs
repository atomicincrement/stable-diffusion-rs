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
    } else {
        print_help();
    }
}

fn print_help() {
    println!("Stable Diffusion Demo - Text to Image Generation");
    println!();
    println!("Usage:");
    println!("  cargo run -- download    Download pretrained weights");
    println!("  cargo run -- test        Test weight loading");
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
    
    // Check if weights exist locally
    let weight_path = "./weights/runwayml_stable-diffusion-v1-5.safetensors";
    
    match weights::WeightStore::load_from_safetensors(weight_path) {
        Ok(_) => println!("✓ Weights loaded successfully from {}", weight_path),
        Err(e) => {
            println!("✗ Could not load weights: {}", e);
            println!();
            println!("To test weight loading:");
            println!("1. Run: cargo run -- download");
            println!("2. Then: cargo run -- test");
        }
    }
}
