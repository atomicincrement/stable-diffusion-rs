# Noise & UNet Comparison Summary

## Findings

### NOISE SCHEDULES

#### Our Implementation
- **Linear**: β ∈ [0.0001, 0.02]
- **Cosine**: Uses standard formula ᾱ_t = cos²(πt/2000)

#### Hugging Face (SD v1.5)
- **ScaledLinear**: β ∈ [0.00085, 0.012] with sqrt interpolation
- **SquaredcosCapV2**: Uses offset formula ᾱ_t = cos²((t+0.008)/1.008 * π/2)

#### Key Differences
| Aspect | Ours | HF |
|--------|------|-----|
| Beta Start | 0.0001 | 0.00085 |
| Beta End | 0.02 | 0.012 |
| Min Noise | 10x Higher | Conservative |
| Interpolation | Linear | Square-root (scaled) |

**Impact**: Our schedule adds noise 10x faster, which explains why output is degraded (too much noise in early steps).

### UNet ARCHITECTURE

#### Our Stub Implementation
- ❌ Returns input unchanged
- ❌ No actual convolutions
- ❌ Cross-attention hardcoded to 768 dim
- ❌ No skip connections
- ❌ No group normalization

#### Hugging Face Implementation
- ✅ Full 2D convolutions with proper channels
- ✅ 4 downsampling blocks → mid block → 3 upsampling blocks
- ✅ Cross-attention properly projects 768 → **1280**
- ✅ Skip connections between corresponding blocks
- ✅ Group normalization (32 groups, eps=1e-5)

#### Block Configuration (HF)
```
Input: 4 channels (latent)
  ↓
Conv2d: 4 → 320 channels
  ↓
Down Block 0: 320 channels (with cross-attn)
  ↓
Down Block 1: 640 channels (with cross-attn)
  ↓
Down Block 2: 1280 channels (with cross-attn)
  ↓
Down Block 3: 1280 channels (NO cross-attn)
  ↓
Mid Block: 1280 channels (with cross-attn)
  ↓
Up Block 0: 1280 channels (with cross-attn)
  ↓
Up Block 1: 640 channels (with cross-attn)
  ↓
Up Block 2: 320 channels (with cross-attn)
  ↓
Conv2d: 1280 → 4 channels
  ↓
Output: 4 channels (denoised latent)
```

#### ⚠️ CRITICAL: CLIP Output Dimension Mismatch
- **CLIP outputs**: (77, 768)
- **UNet expects**: (77, 1280)
- **Solution**: CLIP encoder output needs **projection layer** (768 → 1280)

This is a fundamental architecture issue - the models must be properly connected!

### VAE Architecture

#### Current Implementation
- Simple nearest-neighbor upsampling (8×)
- No learnable decoder blocks
- Output: (1, 3, 512, 512) but just blurred noise

#### Reference Implementation
- Proper decoder blocks with convolutions
- Learnable upsampling (transpose conv or learned upsampling)
- Group normalization
- Residual connections

---

## Why We Generate Wrong Images: Root Causes

1. **Noise schedule is too aggressive**
   - Adding 10x more noise per step
   - Overshooting clean images early

2. **UNet doesn't denoise**
   - Returns zeros/input unchanged
   - Diffusion loop gets no guidance

3. **CLIP→UNet dimension mismatch**
   - 768 ≠ 1280 → can't use embeddings directly
   - Need projection layer

4. **VAE is a placeholder**
   - No learned decoder
   - Just upsampling noise

---

## Quick Fix Priority

### High Priority (Breaking Issues)
1. [ ] **Fix noise schedule** (use HF parameters)
   - beta_start: 0.00085
   - beta_end: 0.012
   - Use ScaledLinear (sqrt interpolation)

2. [ ] **Add CLIP→UNet projection**
   - Linear layer: 768 → 1280
   - Or integrate into encoder

3. [ ] **Load UNet weights properly**
   - Parse 686 tensors from safetensors
   - Map to conv/norm/attention layers

### Medium Priority (Functional Requirements)
4. [ ] **Implement 2D convolution**
   - Can use ndarray or simple kernel ops
   - Group normalization support

5. [ ] **Add skip connections**
   - Connect down-block outputs to up-blocks

6. [ ] **Fix timestep embedding**
   - Broadcast to spatial dimensions properly

### Low Priority (Polish)
7. [ ] Implement proper VAE decoder
8. [ ] Add classifier-free guidance
9. [ ] Optimize tensor operations

---

## Quick Debugging Test

```bash
# Test our noise schedule
cargo run --release -- noise-test

# Compare with reference values
cargo run --release --example compare_noise

# Analyze CLIP weights
cargo run --release --example debug_clip
```

---

## Next Steps

1. **Apply noise schedule fix immediately**
   - Expected improvement: Better diffusion starting point

2. **Debug CLIP→UNet mismatch**
   - Add projection layer or encoder output scaling

3. **Load actual UNet weights**
   - Map 686 tensors to model parameters
   - Test shapes

4. **Run reference comparison**
   - Use Candle to verify intermediate outputs
   - Compare attention maps, residuals, etc.

---

## Architecture Complexity

Candle's UNet has ~686 weight tensors because:
- 4 down blocks × (2 residual blocks + attention) × layers = ~200 tensors
- 1 mid block = ~100 tensors  
- 3 up blocks × (2 residual blocks + attention + skip) × layers = ~300 tensors
- Input/output convolutions = ~10 tensors
- Various norms and embeddings = ~76 tensors

We're implementing this incrementally, starting with the foundation.
