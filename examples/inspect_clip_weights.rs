use std::fs::File;
use safetensors::SafeTensors;
use memmap2::Mmap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "./weights/runwayml_stable-diffusion-v1-5/text_encoder/model.safetensors";
    let file = File::open(file_path)?;
    
    // Memory map the file
    let mmap = unsafe { Mmap::map(&file)? };
    
    // Parse safetensors format
    let tensors = SafeTensors::deserialize(&mmap)?;
    
    println!("{}", "=".repeat(100));
    println!("CLIP TEXT ENCODER - TENSOR STRUCTURE");
    println!("{}", "=".repeat(100));
    println!();
    
    let mut tensor_list: Vec<(String, Vec<usize>, String)> = Vec::new();
    
    for (name, tensor_view) in tensors.tensors() {
        let shape = tensor_view.shape().to_vec();
        let dtype = format!("{:?}", tensor_view.dtype());
        tensor_list.push((name.clone(), shape, dtype));
    }
    tensor_list.sort_by(|a, b| a.0.cmp(&b.0));
    
    let mut total_params = 0u64;
    
    for (name, shape, dtype) in &tensor_list {
        
        let mut num_params = 1u64;
        for &dim in shape {
            num_params *= dim as u64;
        }
        total_params += num_params;
        
        // Estimate size based on dtype
        let bytes_per_param = if dtype.contains("F32") {
            4
        } else if dtype.contains("F16") || dtype.contains("BF16") {
            2
        } else {
            4
        };
        let size_mb = (num_params as f64 * bytes_per_param as f64) / (1024.0 * 1024.0);
        
        // Format shape as string
        let shape_str = format!("{:?}", shape);
        
        println!("{:60} {:35} {:10} {:8.2} MB", name, shape_str, dtype, size_mb);
    }
    
    println!();
    println!("{}", "=".repeat(100));
    println!("TOTAL TENSORS: {}", tensor_list.len());
    println!("TOTAL PARAMETERS: {} ({:.2} M)", total_params, total_params as f64 / 1_000_000.0);
    println!("ESTIMATED TOTAL SIZE: {:.2} MB", (total_params as f64 * 4.0) / (1024.0 * 1024.0));
    println!("{}", "=".repeat(100));
    
    // Analyze structure
    println!("\n--- STRUCTURE ANALYSIS ---\n");
    
    let mut groups: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();
    for (name, _, _) in &tensor_list {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() > 0 {
            let group = if parts[0].contains("embeddings") {
                "EMBEDDINGS".to_string()
            } else if parts[0].contains("encoder") {
                "TRANSFORMER ENCODER".to_string()
            } else if parts[0].contains("final") {
                "FINAL LAYER NORM & PROJECTION".to_string()
            } else {
                format!("GROUP: {}", parts[0])
            };
            
            groups.entry(group).or_insert_with(Vec::new).push(name.to_string());
        }
    }
    
    for (group, tensors) in groups {
        println!("{}:", group);
        for tensor_name in tensors {
            println!("  • {}", tensor_name);
        }
        println!();
    }
    
    Ok(())
}
