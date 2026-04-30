# Debugging: Comparing with Hugging Face Candle

## Goal
Compare our Stable Diffusion implementation with Hugging Face Candle's reference implementation to identify where the wrong images are coming from.

## Setup
- Reference implementation cloned to `.hf/candle/`
- Candle Stable Diffusion example built and ready
- Both using same model: `runwayml/stable-diffusion-v1-5`
- Test prompt: "a cat on a beach"

## Testing Plan

### 1. Test CLIP Text Encoder Output
- Run both implementations with same prompt
- Compare (77, 768) embeddings
- Verify tokenization, attention, MLP computations

Our implementation:
```bash
cargo run --release -- generate --prompt "a cat on a beach" --steps 1 --output /tmp/ours_clip_test.png
```

Candle reference:
```bash
cd .hf/candle && cargo run --example stable-diffusion --release --cpu -- --prompt "a cat on a beach" --n-steps 1 --final-image /tmp/candle_clip_test.png
```

### 2. Test Noise Schedule
- Verify cosine schedule values match
- Check timestep embeddings

Our noise schedule values:
```bash
cargo run --release -- noise-test
```

### 3. Test Diffusion Steps
- Run both with same seed (if possible)
- Compare latent outputs at key timesteps
- Check if UNet predictions are reasonable

### 4. Test VAE Decoder
- Compare upsampling logic
- Check channel projections (4 → 3)
- Verify output range [0, 1]

## Key Comparison Points

1. **CLIP Embeddings**: Should have reasonable range and distribution
2. **Noise Schedule**: Linear/cosine values should match exactly
3. **Timestep Embeddings**: Should encode time information properly
4. **UNet Output**: Should predict noise, not generate garbage
5. **VAE Upsampling**: Should properly upsample (64,64) → (512,512)

## Debugging Steps

1. Extract CLIP embeddings from both, save as tensor
2. Compute statistics (mean, std, min, max)
3. Compare attention weights
4. Compare diffusion latent at each step
5. Visualize intermediate latents
