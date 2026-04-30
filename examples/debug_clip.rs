// Debug program to compare CLIP embeddings

use std::fs::File;
use memmap2::Mmap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load CLIP model
    let clip_path = "./weights/runwayml_stable-diffusion-v1-5/text_encoder/model.safetensors";
    
    println!("Loading CLIP from: {}", clip_path);
    
    let file = File::open(clip_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    
    // Print available tensors
    println!("\nAvailable tensors in CLIP:");
    for name in tensors.names() {
        if let Ok(tensor) = tensors.tensor(name) {
            println!("  {:50} | shape: {:?} | dtype: {:?}", 
                     name, 
                     tensor.shape(), 
                     tensor.dtype());
        }
    }
    
    // Load token embeddings
    println!("\n\nToken Embeddings Statistics:");
    if let Ok(token_emb) = tensors.tensor("text_model.embeddings.token_embedding.weight") {
        println!("  Shape: {:?}", token_emb.shape());
        println!("  DType: {:?}", token_emb.dtype());
        
        // Convert to f32 for analysis
        let data = token_emb.data();
        let f32_data: Vec<f32> = if token_emb.dtype() == safetensors::Dtype::F32 {
            // Safe cast since we know it's f32
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4).to_vec()
            }
        } else {
            vec![]
        };
        
        if !f32_data.is_empty() {
            let min = f32_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = f32_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = f32_data.iter().sum::<f32>() / f32_data.len() as f32;
            
            println!("  Min: {:.6}", min);
            println!("  Max: {:.6}", max);
            println!("  Mean: {:.6}", mean);
            println!("  First 10 values: {:?}", &f32_data[..10.min(f32_data.len())]);
        }
    }
    
    // Load position embeddings
    println!("\n\nPosition Embeddings Statistics:");
    if let Ok(pos_emb) = tensors.tensor("text_model.embeddings.position_embedding.weight") {
        println!("  Shape: {:?}", pos_emb.shape());
        println!("  DType: {:?}", pos_emb.dtype());
    }
    
    println!("\nDone!");
    
    Ok(())
}
