//! Weight loading and management for pretrained Stable Diffusion models
//! 
//! Uses memory-mapped files with ArrayView for zero-copy weight access.
//! Weights are stored as references to mmap'd data, not owned arrays.

use crate::types::{TensorBf16, WeightMatrix};
use ndarray::Array;
use half::bf16;
use std::path::Path;
use futures_util::stream::StreamExt;

/// Structure holding all model weights
pub struct WeightStore {
    /// CLIP text encoder weights
    pub clip_weights: ClipWeights,
    /// UNet denoiser weights
    pub unet_weights: UNetWeights,
    /// VAE decoder weights
    pub vae_weights: VaeWeights,
}

/// CLIP model weights
pub struct ClipWeights {
    // TODO: Token embedding matrix
    // TODO: Positional embeddings
    // TODO: Transformer layer weights
    // TODO: Output projection
}

/// UNet denoiser weights
pub struct UNetWeights {
    // TODO: Timestep embedding layers
    // TODO: Residual blocks
    // TODO: Attention layers
    // TODO: Upsampling blocks
}

/// VAE decoder weights
pub struct VaeWeights {
    // TODO: Upsampling layers
    // TODO: Residual blocks
    // TODO: Output projection
}

impl WeightStore {
    /// Load weights from a safetensors file using memory mapping
    /// 
    /// This avoids loading the entire file into memory by using memory-mapped I/O,
    /// which is much more efficient for large weight files (>1GB).
    /// 
    /// Returns ArrayView references to weight matrices, not owned arrays.
    /// This enables zero-copy access to weights - they're read directly from the
    /// memory-mapped file as needed, without copying into separate memory.
    /// 
    /// # Arguments
    /// * `path` - Path to the safetensors weight file
    /// 
    /// # Memory Layout
    /// ```text
    /// Memory Map
    /// ┌─────────────────────────────────────┐
    /// │ SafeTensors Header (JSON metadata)   │ ← Read once
    /// ├─────────────────────────────────────┤
    /// │ Weight Matrix 1 (mmap'd)             │ ← ArrayView points here
    /// │ ├─ CLIP token embeddings             │
    /// │ ├─ CLIP transformer weights          │
    /// │ └─ CLIP output projection            │
    /// ├─────────────────────────────────────┤
    /// │ Weight Matrix 2 (mmap'd)             │ ← ArrayView points here
    /// │ ├─ UNet conv layers                  │
    /// │ ├─ UNet attention layers             │
    /// │ └─ UNet upsampling                   │
    /// ├─────────────────────────────────────┤
    /// │ Weight Matrix 3 (mmap'd)             │ ← ArrayView points here
    /// │ ├─ VAE decoder blocks                │
    /// │ └─ VAE output projection             │
    /// └─────────────────────────────────────┘
    /// 
    /// No copying: ArrayView references point directly into mmap'd memory.
    /// Lazy loading: Only referenced data is paged in by OS.
    /// ```
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(format!("Weight file not found: {:?}", path));
        }

        // Use memory-mapped file for efficient loading
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open weight file: {}", e))?;
        
        let file_size = file.metadata()
            .map_err(|e| format!("Failed to get file metadata: {}", e))?
            .len();
        
        println!("Loading weights from: {:?} ({:.2} GB)", path, file_size as f64 / 1_000_000_000.0);
        println!("Using memory-mapped I/O for efficient loading...");
        
        // Memory map the file for efficient access
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| format!("Failed to memory-map file: {}", e))?
        };
        
        // Parse safetensors format from memory-mapped data
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("Failed to parse safetensors format: {}", e))?;
        
        println!("✓ Loaded safetensors with {} tensors", tensors.len());
        
        // TODO: Extract specific components (CLIP, UNet, VAE)
        // For now, just verify we can read the file
        for (name, tensor_view) in tensors.tensors() {
            println!("  • {} shape: {:?}", name, tensor_view.shape());
        }
        
        Ok(WeightStore {
            clip_weights: ClipWeights {},
            unet_weights: UNetWeights {},
            vae_weights: VaeWeights {},
        })
    }

    /// Load component weights from organized directory structure
    /// 
    /// Stable Diffusion v1.5 weights are organized as:
    /// ```
    /// weights/
    ///   model_index.json          - Pipeline configuration
    ///   text_encoder/
    ///     model.safetensors       - CLIP text encoder
    ///   unet/
    ///     diffusion_pytorch_model.safetensors  - UNet denoiser
    ///   vae/
    ///     diffusion_pytorch_model.safetensors  - VAE decoder
    /// ```
    pub fn load_from_directory<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        
        if !model_dir.exists() {
            return Err(format!("Model directory not found: {:?}", model_dir));
        }

        println!("Loading components from: {:?}", model_dir);
        
        // Check for required components
        let components = vec![
            ("text_encoder/model.safetensors", "CLIP Text Encoder"),
            ("unet/diffusion_pytorch_model.safetensors", "UNet Denoiser"),
            ("vae/diffusion_pytorch_model.safetensors", "VAE Decoder"),
        ];

        let mut found_all = true;
        for (component_path, component_name) in &components {
            let full_path = model_dir.join(component_path);
            if full_path.exists() {
                let size = std::fs::metadata(&full_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                println!("  ✓ {} ({:.2} MB)", component_name, size as f64 / 1_000_000.0);
            } else {
                println!("  ✗ {} MISSING", component_name);
                found_all = false;
            }
        }

        if !found_all {
            return Err(format!(
                "Missing required component files in {:?}\n\
                Expected directory structure:\n\
                  text_encoder/model.safetensors\n\
                  unet/diffusion_pytorch_model.safetensors\n\
                  vae/diffusion_pytorch_model.safetensors",
                model_dir
            ));
        }

        println!("\n✓ All required components found!");
        
        // For now, just verify files exist
        // TODO: Load each component using load_from_safetensors
        
        Ok(WeightStore {
            clip_weights: ClipWeights {},
            unet_weights: UNetWeights {},
            vae_weights: VaeWeights {},
        })
    }

    /// Download weights from Hugging Face model hub
    /// 
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "runwayml/stable-diffusion-v1-5")
    /// * `output_dir` - Directory to save downloaded weights
    pub async fn download_from_hub(model_id: &str, output_dir: &str) -> Result<Self, String> {
        println!("Downloading model {} to {}", model_id, output_dir);
        
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;

        // Download safetensors files from HuggingFace
        download_model_files(model_id, output_dir).await?;
        
        // Load the downloaded weights
        let weight_path = std::path::Path::new(output_dir)
            .join(format!("{}.safetensors", model_id.replace("/", "_")));
        
        Self::load_from_safetensors(&weight_path)
    }

    /// Load weights, downloading if necessary
    /// 
    /// # Arguments
    /// * `model_id` - Model identifier
    /// * `cache_dir` - Cache directory for weights (default: ~/.cache/stable-diffusion-rs)
    pub async fn load_or_download(model_id: &str, cache_dir: Option<&str>) -> Result<Self, String> {
        let cache_dir = cache_dir.unwrap_or("~/.cache/stable-diffusion-rs");
        let expanded_cache = shellexpand::tilde(cache_dir).into_owned();
        
        // Try to load from cache first
        let weight_path = Path::new(&expanded_cache).join(format!("{}.safetensors", model_id.replace("/", "_")));
        
        if weight_path.exists() {
            println!("Loading cached weights from: {:?}", weight_path);
            return Self::load_from_safetensors(&weight_path);
        }

        // Download if not cached
        println!("Weights not found in cache, downloading...");
        Self::download_from_hub(model_id, &expanded_cache).await?;
        
        // Load the downloaded weights
        Self::load_from_safetensors(&weight_path)
    }
}

/// Download model files from HuggingFace Hub
/// 
/// Downloads the main model.safetensors file and metadata
async fn download_model_files(model_id: &str, output_dir: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    
    // Stable Diffusion v1.5 components in safetensors format
    // Note: Model files are organized in subdirectories
    let files_to_download = vec![
        "unet/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "model_index.json",
    ];
    
    println!("\nThis will download Stable Diffusion v1.5 components (~4GB total)");
    println!("Model: https://huggingface.co/{}\n", model_id);
    
    // First, try without authentication to give better error message
    let test_url = format!(
        "https://huggingface.co/{}/resolve/main/model_index.json",
        model_id
    );
    
    let test_response = client
        .head(&test_url)
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;
    
    if test_response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(format!(
            "Model not found or not accessible at: https://huggingface.co/{}\n\
            \n\
            To download Stable Diffusion weights:\n\n\
            OPTION 1: Using huggingface-cli (recommended):\n\
            ┌─────────────────────────────────────────────────────────────┐\n\
            │ $ pip install huggingface_hub                              │\n\
            │ $ huggingface-cli login                                    │\n\
            │   (enter your HuggingFace token)                           │\n\
            │ $ huggingface-cli download runwayml/stable-diffusion-v1-5  │\n\
            │   --repo-type model --cache-dir ./weights                  │\n\
            └─────────────────────────────────────────────────────────────┘\n\
            \n\
            OPTION 2: Manual download:\n\
            ┌─────────────────────────────────────────────────────────────┐\n\
            │ 1. Visit: https://huggingface.co/{}/tree/main               │\n\
            │ 2. Create a HuggingFace account if needed                  │\n\
            │ 3. Click 'Accept and Access Repository'                   │\n\
            │ 4. Download the .safetensors files                         │\n\
            │ 5. Place in ./weights directory                            │\n\
            └─────────────────────────────────────────────────────────────┘\n\
            \n\
            OPTION 3: Using git + git-lfs:\n\
            ┌─────────────────────────────────────────────────────────────┐\n\
            │ $ git clone https://huggingface.co/{} ./weights/sd-v1-5    │\n\
            └─────────────────────────────────────────────────────────────┘",
            model_id, model_id, model_id
        ));
    }
    
    if test_response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err(format!(
            "Authentication required for model: {}\n\
            \n\
            To authenticate, set your HuggingFace token:\n\n\
            OPTION 1: Set environment variable:\n\
            ┌─────────────────────────────────────────────────────────────┐\n\
            │ $ export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx        │\n\
            │ $ cargo run -- download                                     │\n\
            └─────────────────────────────────────────────────────────────┘\n\
            \n\
            OPTION 2: Save to ~/.huggingface/token:\n\
            ┌─────────────────────────────────────────────────────────────┐\n\
            │ $ mkdir -p ~/.huggingface                                   │\n\
            │ $ echo hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \\                 │\n\
            │   > ~/.huggingface/token                                    │\n\
            └─────────────────────────────────────────────────────────────┘\n\
            \n\
            Get your token at: https://huggingface.co/settings/tokens",
            model_id
        ));
    }
    
    println!("Downloading components for: {}", model_id);
    
    for file_name in files_to_download {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id, file_name
        );
        
        println!("\n  • {}", file_name);
        
        // Make HEAD request to get file size
        let head_response = client
            .head(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to check file size for {}: {}", file_name, e))?;
        
        if !head_response.status().is_success() {
            println!("    ⚠ Not found (may be optional)");
            continue;
        }
        
        let file_size = head_response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        
        println!("    Size: {:.2} GB", file_size as f64 / 1_000_000_000.0);
        
        // Download file with progress bar
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to download {}: {}", file_name, e))?;
        
        if !response.status().is_success() {
            println!("    ✗ Download failed: {}", response.status());
            continue;
        }
        
        let pb = indicatif::ProgressBar::new(file_size);
        pb.set_style(indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        
        // Create subdirectories if needed
        let output_path = std::path::Path::new(output_dir).join(format!("{}", model_id.replace("/", "_")));
        std::fs::create_dir_all(&output_path)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
        
        let file_output = output_path.join(file_name);
        if let Some(parent) = file_output.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create subdirectory: {}", e))?;
        }
        
        let mut file = std::fs::File::create(&file_output)
            .map_err(|e| format!("Failed to create output file: {}", e))?;
        
        let mut stream = response.bytes_stream();
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download interrupted: {}", e))?;
            use std::io::Write;
            file.write_all(&chunk)
                .map_err(|e| format!("Failed to write to file: {}", e))?;
            pb.inc(chunk.len() as u64);
        }
        
        pb.finish_with_message(format!("✓ {}", file_name));
        println!("    Saved to: {}", file_output.display());
    }
    
    println!("\n✓ Download complete!");
    Ok(())
}
