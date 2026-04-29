//! CLIP Input/Output Demonstration
//! 
//! This example demonstrates the expected data flow through CLIP:
//! Input text → Tokenization → Token IDs → Embedding → (77, 768) output
//! 
//! Shows typical shapes and values at each stage.

use ndarray::{Array1, Array2, s};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CLIP TEXT ENCODER - Input/Output Data Flow");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Stage 1: Input text
    let prompts = vec![
        "a cat on a beach",
        "a beautiful sunset over the ocean",
        "dog",
    ];

    for prompt in prompts {
        println!("📝 INPUT PROMPT:");
        println!("   Text: \"{}\"", prompt);
        println!("   Length: {} characters\n", prompt.len());

        // Stage 2: Tokenization (simulated)
        let tokens = simulate_tokenization(prompt);
        println!("🔢 TOKENIZATION:");
        println!("   Raw token count: {}", tokens.len());
        println!("   Tokens (first 10): {:?}", &tokens[..tokens.len().min(10)]);
        
        // Stage 3: Pad/truncate to 77 tokens
        let token_ids = pad_to_77(&tokens);
        println!("\n⏲️  PADDING TO 77 TOKENS:");
        println!("   Padded token IDs shape: ({})",  token_ids.len());
        println!("   Padded tokens (first 10): {:?}", token_ids.slice(s![..10]).to_vec());
        println!("   Padded tokens (last 5): {:?}", token_ids.slice(s![72..]).to_vec());

        // Stage 4: Token Embedding lookup
        let token_embeddings = simulate_token_embeddings(&token_ids);
        println!("\n🔗 TOKEN EMBEDDING LOOKUP:");
        println!("   Shape: {:?}", token_embeddings.dim());
        println!("   Dtype: f32");
        println!("   Range: [{:.6}, {:.6}]", 
            token_embeddings.iter().cloned().fold(f32::INFINITY, f32::min),
            token_embeddings.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Stage 5: Add positional embeddings
        let with_pos = simulate_positional_encoding(&token_embeddings);
        println!("\n➕ POSITIONAL ENCODING:");
        println!("   Shape: {:?}", with_pos.dim());
        println!("   After adding position embeddings");

        // Stage 6: Transformer processing
        let after_transformer = simulate_transformer_blocks(&with_pos);
        println!("\n🔄 TRANSFORMER BLOCKS (12 layers):");
        println!("   Input shape:  {:?}", with_pos.dim());
        println!("   Output shape: {:?}", after_transformer.dim());
        println!("   Each block: LayerNorm → Attention → LayerNorm → MLP → Add/Norm");

        // Stage 7: Final layer norm
        let output = simulate_final_norm(&after_transformer);
        println!("\n🎯 FINAL OUTPUT (after LayerNorm):");
        println!("   Shape: {:?}", output.dim());
        println!("   Dtype: f32");
        println!("   Statistics:");
        println!("     Mean: {:.6}", output.iter().sum::<f32>() / output.len() as f32);
        let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("     Min: {:.6}", min);
        println!("     Max: {:.6}", max);
        println!("     Range: {:.6}", max - min);

        // Stage 8: Use in diffusion
        println!("\n📤 USE IN DIFFUSION MODEL:");
        println!("   This (77, 768) embedding is used as cross-attention conditioning");
        println!("   in the UNet denoiser: UNet(noise, timestep, text_embedding)");
        println!("   ✓ Ready for generation\n");
    }

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Input:  Text string (variable length)");
    println!("Output: Tensor of shape (77, 768)");
    println!("        - 77 token positions");
    println!("        - 768 dimensions per token");
    println!("        - f32 precision");
    println!("        - Ready for cross-attention in diffusion model");
    println!("\nArchitecture:");
    println!("  • Vocabulary: 49,408 tokens");
    println!("  • Embedding dim: 768");
    println!("  • Transformer layers: 12");
    println!("  • Attention heads: 12");
    println!("  • Total params: 123.06M");
}

/// Simulate tokenization (simplified - uses fixed tokens for demo)
fn simulate_tokenization(text: &str) -> Vec<u32> {
    // In real implementation, use tokenizers crate
    // For demo, create deterministic tokens from text length
    let mut tokens = Vec::new();
    
    // Start token
    tokens.push(49406);
    
    // Text tokens (simulate by splitting words)
    for word in text.split_whitespace() {
        let hash = word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        tokens.push(hash % 49400 + 1); // Keep in valid range [1, 49400]
    }
    
    // End token
    tokens.push(49407);
    
    tokens
}

/// Pad token IDs to exactly 77 tokens
fn pad_to_77(tokens: &[u32]) -> Array1<u32> {
    let mut padded = Array1::zeros(77);
    let copy_len = tokens.len().min(77);
    padded.slice_mut(s![..copy_len]).assign(&Array1::from_vec(tokens[..copy_len].to_vec()));
    // Remaining positions are 0 (padding token)
    padded
}

/// Simulate token embedding lookup
/// Returns shape (77, 768)
fn simulate_token_embeddings(token_ids: &Array1<u32>) -> Array2<f32> {
    let mut embeddings = Array2::zeros((77, 768));
    
    // Simulate: each token gets a unique embedding based on its ID
    for (i, &token_id) in token_ids.iter().enumerate() {
        for j in 0..768 {
            // Deterministic but varied embeddings
            let val = ((token_id as f32 + 1.0) * (j as f32 + 1.0)).sin() * 0.1;
            embeddings[[i, j]] = val;
        }
    }
    
    embeddings
}

/// Simulate positional encoding addition
/// Returns shape (77, 768)
fn simulate_positional_encoding(token_embeddings: &Array2<f32>) -> Array2<f32> {
    let mut result = token_embeddings.clone();
    
    // Add positional encoding (sinusoidal pattern)
    for i in 0..77 {
        for j in 0..768 {
            let pos_val = (i as f32 / 77.0) * (j as f32 / 768.0) * std::f32::consts::PI;
            result[[i, j]] += pos_val.sin() * 0.01;
        }
    }
    
    result
}

/// Simulate transformer blocks (12 layers)
/// Each layer: LayerNorm → Attention → LayerNorm → MLP
/// Returns shape (77, 768)
fn simulate_transformer_blocks(input: &Array2<f32>) -> Array2<f32> {
    let mut x = input.clone();
    
    // Simplified: 12 transformer blocks
    for _layer in 0..12 {
        // Simulate layer processing (apply small transformations)
        x = x.clone() * 0.99; // Slight decay to simulate processing
        x = x + 0.001; // Add small bias
    }
    
    x
}

/// Simulate final layer normalization
/// Returns shape (77, 768)
fn simulate_final_norm(input: &Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros(input.dim());
    
    // Layer norm per token (normalize across dimensions)
    for i in 0..77 {
        let token_row = input.row(i);
        let mean = token_row.iter().sum::<f32>() / 768.0;
        let variance = token_row.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / 768.0;
        let std = (variance + 1e-5).sqrt();
        
        for j in 0..768 {
            output[[i, j]] = (input[[i, j]] - mean) / std;
        }
    }
    
    output
}
