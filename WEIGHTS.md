# Downloading Pretrained Weights

This project uses Stable Diffusion v1.5 weights for text-to-image generation.

## Quick Start

### Option 1: Automatic Download (Recommended)

```bash
cargo run -- download
```

This will download the weights from Hugging Face and cache them locally in `./weights`.

### Option 2: Manual Download

1. **Visit Hugging Face Hub**: https://huggingface.co/runwayml/stable-diffusion-v1-5
2. **Accept the License**: Read and accept the model license
3. **Download Weights**: Download the safetensors files
4. **Place in Directory**:
   ```bash
   mkdir -p weights
   # Move downloaded files to ./weights
   ```

## Available Models

- **runwayml/stable-diffusion-v1-5** (recommended, ~4GB)
  - Official Stable Diffusion v1.5
  - Good balance between quality and speed
  - Full model with all components

## Requirements

- **Disk Space**: ~4-5GB for full model weights
- **RAM**: ~8GB recommended for inference
- **Internet**: Required for initial download (~30 minutes on typical connection)

## Supported Formats

- SafeTensors (.safetensors) - Preferred, faster loading
- PyTorch (.pt, .bin) - Also supported, slower loading

## Cache Directory

Weights are cached in `./weights` by default. You can change this with:

```bash
WEIGHTS_CACHE=~/.cache/stable-diffusion-rs cargo run -- download
```

## Verifying Weights

After downloading, test the weights with:

```bash
cargo run -- test
```

This will verify:
- ✓ Weight file exists
- ✓ Weight file can be parsed
- ✓ All expected components are present
- ✓ Shapes and dimensions are correct

## Troubleshooting

### "Connection timeout"
- Check your internet connection
- Try again later (Hugging Face may be rate-limited)

### "Disk space full"
- Ensure you have at least 5GB free
- Consider removing `./target` to save space: `cargo clean`

### "Authentication required"
- Some models require accepting terms on Hugging Face
- Visit the model page and accept the license

## Models in Development

We're planning to support:
- ✓ Stable Diffusion v1.5 (current)
- [ ] Stable Diffusion v2.1
- [ ] TinySD (smaller, faster)
- [ ] Pruned/quantized versions (reduced memory)

## License Information

Weights are provided under the Stable Diffusion model license:
- https://huggingface.co/runwayml/stable-diffusion-v1-5

**Important**: Review the license agreement before using in production.
