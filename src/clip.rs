//! CLIP text encoder for generating text embeddings

use crate::types::{CLIP_EMBEDDING_DIM, MAX_TOKEN_LENGTH, CLIP_NUM_LAYERS, CLIP_NUM_HEADS, CLIP_MLP_EXPANSION, TOKEN_VOCAB_SIZE};
use ndarray::{Array1, Array2, Axis, s};

/// CLIP text encoder model with weights
pub struct ClipEncoder {
    /// Token embeddings [vocab_size, embed_dim] = [49408, 768]
    token_embeddings: Array2<f32>,
    /// Position embeddings [max_seq_length, embed_dim] = [77, 768]
    position_embeddings: Array2<f32>,
    /// Transformer layer weights
    transformer_layers: Vec<TransformerLayer>,
    /// Final layer normalization
    final_norm_weight: Array1<f32>,
    final_norm_bias: Array1<f32>,
}

/// Single transformer layer weights
struct TransformerLayer {
    /// Layer norm 1
    norm1_weight: Array1<f32>,
    norm1_bias: Array1<f32>,
    /// Self-attention
    q_proj: Array2<f32>,
    k_proj: Array2<f32>,
    v_proj: Array2<f32>,
    out_proj: Array2<f32>,
    /// Layer norm 2
    norm2_weight: Array1<f32>,
    norm2_bias: Array1<f32>,
    /// MLP
    fc1_weight: Array2<f32>,  // [embed_dim, mlp_dim]
    fc1_bias: Array1<f32>,
    fc2_weight: Array2<f32>,  // [mlp_dim, embed_dim]
    fc2_bias: Array1<f32>,
}

impl ClipEncoder {
    /// Create a new CLIP encoder from weight file
    /// 
    /// Loads weights from safetensors file and initializes the model.
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        println!("Loading CLIP encoder from: {}", path);
        
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open CLIP weights: {}", e))?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| format!("Failed to memory-map CLIP file: {}", e))?
        };

        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("Failed to parse CLIP safetensors: {}", e))?;

        // Load token embeddings
        let token_embeddings = load_tensor_2d(&tensors, "text_model.embeddings.token_embedding.weight")?;
        println!("  ✓ Token embeddings: {:?}", token_embeddings.dim());

        // Load position embeddings
        let position_embeddings = load_tensor_2d(&tensors, "text_model.embeddings.position_embedding.weight")?;
        println!("  ✓ Position embeddings: {:?}", position_embeddings.dim());

        // Load transformer layers
        let mut transformer_layers = Vec::new();
        for layer_idx in 0..CLIP_NUM_LAYERS {
            let layer = TransformerLayer::load(&tensors, layer_idx)?;
            transformer_layers.push(layer);
        }
        println!("  ✓ Loaded {} transformer layers", CLIP_NUM_LAYERS);

        // Load final layer norm
        let final_norm_weight = load_tensor_1d(&tensors, "text_model.final_layer_norm.weight")?;
        let final_norm_bias = load_tensor_1d(&tensors, "text_model.final_layer_norm.bias")?;
        println!("  ✓ Final layer norm loaded");

        println!("✅ CLIP encoder loaded successfully!\n");

        Ok(ClipEncoder {
            token_embeddings,
            position_embeddings,
            transformer_layers,
            final_norm_weight,
            final_norm_bias,
        })
    }

    /// Encode text prompt into embedding vector
    /// 
    /// # Arguments
    /// * `text` - Input text prompt (variable length)
    /// 
    /// # Returns
    /// Text embedding of shape (77, 768) ready for diffusion cross-attention
    pub fn encode(&self, text: &str) -> Result<Array2<f32>, String> {
        // Step 1: Tokenize
        let tokens = tokenize(text)?;
        
        // Step 2: Token embeddings
        let mut x = Array2::zeros((MAX_TOKEN_LENGTH, CLIP_EMBEDDING_DIM));
        for (i, &token_id) in tokens.iter().enumerate() {
            if token_id < self.token_embeddings.nrows() {
                x.row_mut(i).assign(&self.token_embeddings.row(token_id));
            }
        }

        // Step 3: Add position embeddings
        for i in 0..MAX_TOKEN_LENGTH {
            x.row_mut(i).zip_mut_with(&self.position_embeddings.row(i), |x_val, pos_val| {
                *x_val = *x_val + *pos_val;
            });
        }

        // Step 4: Transformer blocks
        for layer in &self.transformer_layers {
            x = layer.forward(&x)?;
        }

        // Step 5: Final layer norm
        x = layer_norm(&x, &self.final_norm_weight, &self.final_norm_bias);

        Ok(x)
    }
}

impl TransformerLayer {
    /// Load a single transformer layer from weights
    fn load(tensors: &safetensors::SafeTensors, layer_idx: usize) -> Result<Self, String> {
        let prefix = format!("text_model.encoder.layers.{}", layer_idx);

        Ok(TransformerLayer {
            // Layer norm 1
            norm1_weight: load_tensor_1d(tensors, &format!("{}.layer_norm1.weight", prefix))?,
            norm1_bias: load_tensor_1d(tensors, &format!("{}.layer_norm1.bias", prefix))?,
            // Self-attention
            q_proj: load_tensor_2d(tensors, &format!("{}.self_attn.q_proj.weight", prefix))?,
            k_proj: load_tensor_2d(tensors, &format!("{}.self_attn.k_proj.weight", prefix))?,
            v_proj: load_tensor_2d(tensors, &format!("{}.self_attn.v_proj.weight", prefix))?,
            out_proj: load_tensor_2d(tensors, &format!("{}.self_attn.out_proj.weight", prefix))?,
            // Layer norm 2
            norm2_weight: load_tensor_1d(tensors, &format!("{}.layer_norm2.weight", prefix))?,
            norm2_bias: load_tensor_1d(tensors, &format!("{}.layer_norm2.bias", prefix))?,
            // MLP
            fc1_weight: load_tensor_2d(tensors, &format!("{}.mlp.fc1.weight", prefix))?,
            fc1_bias: load_tensor_1d(tensors, &format!("{}.mlp.fc1.bias", prefix))?,
            fc2_weight: load_tensor_2d(tensors, &format!("{}.mlp.fc2.weight", prefix))?,
            fc2_bias: load_tensor_1d(tensors, &format!("{}.mlp.fc2.bias", prefix))?,
        })
    }

    /// Forward pass through transformer block
    /// Pre-norm architecture: LayerNorm → Attention → Add & Norm → MLP → Add
    /// 
    /// # Arguments
    /// * `x` - Input tensor of shape (77, 768) representing:
    ///   - First 12 blocks: embeddings from previous layer
    ///   - First call (layer 0): token embeddings + positional embeddings
    ///   - Example: (77 sequence positions, 768 embedding dimensions)
    ///   - Each row is a token's embedding vector that will be processed
    ///   - through attention and MLP, with output shape matching input
    fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>, String> {
        // Self-attention branch
        let norm_x = layer_norm(x, &self.norm1_weight, &self.norm1_bias);
        let attn_out = self.multihead_attention(&norm_x)?;
        let x = x + &attn_out; // Residual connection

        // MLP branch
        let norm_x = layer_norm(&x, &self.norm2_weight, &self.norm2_bias);
        
        // MLP: Linear → GELU → Linear
        // fc1: [seq_len, 768] @ [3072, 768]^T → [seq_len, 3072]
        let mut mlp_out = matmul_transpose(&norm_x, &self.fc1_weight);
        for i in 0..mlp_out.nrows() {
            mlp_out.row_mut(i).zip_mut_with(&self.fc1_bias, |out_val, bias_val| {
                *out_val = *out_val + *bias_val;
            });
        }
        
        // GELU activation
        mlp_out = gelu(&mlp_out);

        // fc2: [seq_len, 3072] @ [768, 3072]^T → [seq_len, 768]
        let mut mlp_proj = matmul_transpose(&mlp_out, &self.fc2_weight);
        for i in 0..mlp_proj.nrows() {
            mlp_proj.row_mut(i).zip_mut_with(&self.fc2_bias, |proj_val, bias_val| {
                *proj_val = *proj_val + *bias_val;
            });
        }

        let output = &x + &mlp_proj; // Residual connection

        Ok(output)
    }

    /// Multi-head self-attention
    /// Q, K, V projections → split heads → scaled dot product → merge heads → output projection
    fn multihead_attention(&self, x: &Array2<f32>) -> Result<Array2<f32>, String> {
        let seq_len = x.nrows();
        let head_dim = CLIP_EMBEDDING_DIM / CLIP_NUM_HEADS;

        // Project to Q, K, V: weights stored as [out, in] so use transpose multiply
        let q = matmul_transpose(x, &self.q_proj);
        let k = matmul_transpose(x, &self.k_proj);
        let v = matmul_transpose(x, &self.v_proj);

        // Split heads and compute attention
        let mut attn_out = Array2::zeros((seq_len, CLIP_EMBEDDING_DIM));

        for head_idx in 0..CLIP_NUM_HEADS {
            let start = head_idx * head_dim;
            let end = start + head_dim;

            let q_head = q.slice(s![.., start..end]).to_owned();
            let k_head = k.slice(s![.., start..end]).to_owned();
            let v_head = v.slice(s![.., start..end]).to_owned();

            // Scaled dot-product attention: (Q @ K^T) / sqrt(d_k)
            let scale = 1.0 / (head_dim as f32).sqrt();
            let scores = matmul(&q_head, &k_head.t().to_owned());
            let scores = scores * scale;

            // Softmax
            let scores = softmax(&scores);

            // Apply to values
            let head_out = matmul(&scores, &v_head);

            // Merge back
            attn_out.slice_mut(s![.., start..end]).assign(&head_out);
        }

        // Output projection: weights stored as [out, in] so use transpose multiply
        let output = matmul_transpose(&attn_out, &self.out_proj);

        Ok(output)
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Matrix multiplication: A @ B (standard matrix multiply)
fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    // a is (n, k), b is (k, m), result is (n, m)
    let (n, k) = a.dim();
    let (k2, m) = b.dim();
    assert_eq!(k, k2, "Dimension mismatch in matrix multiplication");

    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[[i, p]] * b[[p, j]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

/// Matrix multiplication: A @ B^T (where B is stored as [out, in] weight matrix)
/// Transforms [seq_len, in_features] @ [out_features, in_features]^T → [seq_len, out_features]
fn matmul_transpose(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    // a is (n, k), b is (m, k) [stored transposed], result is (n, m)
    let (n, k) = a.dim();
    let (m, k2) = b.dim();
    assert_eq!(k, k2, "Dimension mismatch in transposed matrix multiplication");

    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[[i, p]] * b[[j, p]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

/// Tokenize text into token IDs (simplified - uses mock tokenization for now)
fn tokenize(text: &str) -> Result<Array1<usize>, String> {
    // TODO: Use actual tokenizers crate
    // For now, implement mock tokenization
    let mut tokens = vec![49406]; // Start token

    // Simple word-based tokenization (simplified)
    for word in text.split_whitespace() {
        let hash = word.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
        tokens.push(hash % (TOKEN_VOCAB_SIZE - 2) + 1); // Keep in range [1, 49406]
    }

    tokens.push(49407); // End token

    // Pad to MAX_TOKEN_LENGTH
    while tokens.len() < MAX_TOKEN_LENGTH {
        tokens.push(0); // Padding token
    }

    // Truncate if too long
    tokens.truncate(MAX_TOKEN_LENGTH);

    Ok(Array1::from_vec(tokens))
}

/// Load 1D tensor from safetensors
fn load_tensor_1d(tensors: &safetensors::SafeTensors, name: &str) -> Result<Array1<f32>, String> {
    let tensor_view = tensors.tensor(name)
        .map_err(|_| format!("Tensor not found: {}", name))?;

    if tensor_view.shape().len() != 1 {
        return Err(format!("Expected 1D tensor, got {:?}", tensor_view.shape()));
    }

    let data = tensor_view.data();
    let len = tensor_view.shape()[0];
    
    // Convert bytes to f32 array
    let mut result = vec![0f32; len];
    for i in 0..len {
        let offset = i * 4;
        if offset + 4 <= data.len() {
            let bytes = &data[offset..offset + 4];
            result[i] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
    }

    Ok(Array1::from_vec(result))
}

/// Load 2D tensor from safetensors
fn load_tensor_2d(tensors: &safetensors::SafeTensors, name: &str) -> Result<Array2<f32>, String> {
    let tensor_view = tensors.tensor(name)
        .map_err(|_| format!("Tensor not found: {}", name))?;

    if tensor_view.shape().len() != 2 {
        return Err(format!("Expected 2D tensor, got {:?}", tensor_view.shape()));
    }

    let (rows, cols) = (tensor_view.shape()[0], tensor_view.shape()[1]);
    let data = tensor_view.data();
    let mut result = vec![0f32; rows * cols];
    
    for i in 0..rows * cols {
        let offset = i * 4;
        if offset + 4 <= data.len() {
            let bytes = &data[offset..offset + 4];
            result[i] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
    }

    Ok(Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| format!("Failed to create array: {}", e))?)
}

/// GELU activation function
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;

    x.mapv(|val| {
        let cdf = 0.5 * (1.0 + (SQRT_2_OVER_PI * (val + COEFF * val.powi(3))).tanh());
        val * cdf
    })
}

/// Softmax activation (numerically stable)
fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros(x.dim());
    
    for i in 0..x.nrows() {
        let row = x.row(i);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
        
        for j in 0..row.len() {
            result[[i, j]] = ((row[j] - max).exp()) / exp_sum;
        }
    }
    
    result
}

/// Layer normalization
fn layer_norm(x: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>) -> Array2<f32> {
    let mut result = Array2::zeros(x.dim());

    for i in 0..x.nrows() {
        let row = x.row(i);
        let mean = row.mean().unwrap_or(0.0);
        let variance = row.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / row.len() as f32;
        let std = (variance + 1e-5).sqrt();

        for j in 0..row.len() {
            result[[i, j]] = ((row[j] - mean) / std) * gamma[j] + beta[j];
        }
    }

    result
}
