# UNet Architecture Comparison

## Candle UNet 2D Configuration (Stable Diffusion v1.5)

### Input Configuration
```rust
pub struct UNet2DConditionModelConfig {
    pub center_input_sample: bool = false,
    pub flip_sin_to_cos: bool = true,
    pub freq_shift: f64 = 0.0,
    pub layers_per_block: usize = 2,
    pub downsample_padding: usize = 1,
    pub mid_block_scale_factor: f64 = 1.0,
    pub norm_num_groups: usize = 32,
    pub norm_eps: f64 = 1e-5,
    pub cross_attention_dim: usize = 1280,  // ⚠️ NOT 768!
    pub sliced_attention_size: Option<usize> = None,
    pub use_linear_projection: bool = false,
}
```

### Block Configuration
```
Down Blocks (with downsampling):
  ├─ Block 0: out_channels=320,  use_cross_attn=Some(1), attention_head_dim=8
  ├─ Block 1: out_channels=640,  use_cross_attn=Some(1), attention_head_dim=8
  ├─ Block 2: out_channels=1280, use_cross_attn=Some(1), attention_head_dim=8
  └─ Block 3: out_channels=1280, use_cross_attn=None,    attention_head_dim=8 (no attn!)

Mid Block:
  └─ Single block with 1280 channels and cross-attention

Up Blocks (with upsampling):
  ├─ Block 0: out_channels=1280, use_cross_attn=Some(1), attention_head_dim=8
  ├─ Block 1: out_channels=640,  use_cross_attn=Some(1), attention_head_dim=8
  └─ Block 2: out_channels=320,  use_cross_attn=Some(1), attention_head_dim=8
```

### Component Details

1. **Time Embedding**
   - Uses `Timesteps` projection (sinusoidal encoding)
   - Then `TimestepEmbedding` (linear layers)
   - Output dimension: 1280 (matches mid-block)

2. **Conv Layers**
   - Input: Conv2d with 4 → 320 channels
   - Output: Conv2d with 1280 → 4 channels
   - Kernel size: 3×3

3. **Group Normalization**
   - `norm_num_groups: 32`
   - `norm_eps: 1e-5` (default)
   - Applied before attention and after convolutions

4. **Cross-Attention**
   - Query dim: same as block channels (320/640/1280)
   - Context dim: **1280** ← ⚠️ KEY DIFFERENCE!
   - Heads: 8
   - Head dim: channels / 8

### Key Architectural Features

- ✅ Convolutional layers (not fully connected!)
- ✅ Multiple downsampling/upsampling blocks
- ✅ Skip connections between down/up blocks
- ✅ Cross-attention for text conditioning
- ✅ Residual connections
- ✅ Group normalization (not layer norm)

---

## Our Current Implementation Issues

### ❌ Major Issues Found

1. **Cross-Attention Dimension Mismatch**
   - We hardcoded: `context_dim: 768`
   - Should be: `context_dim: 1280` ← **FROM CLIP ENCODER OUTPUT**
   - This means CLIP encoder output needs projection from 768 → 1280!

2. **No Actual Convolutions**
   - We use `ResidualBlock::forward()` which returns input unchanged
   - Should have actual 2D convolutions with proper channels

3. **No Real Skip Connections**
   - No connection between corresponding down/up blocks
   - Skip connections are critical for information flow

4. **Incorrect Timestep Embedding**
   - We create (1280,) embeddings
   - But don't broadcast properly to spatial dimensions

5. **No Proper Normalization**
   - We don't use group norm (32 groups)
   - We don't apply it correctly before/after operations

6. **Cross-Attention Output Wrong**
   - We return shape `(query_dim,)` per timestep
   - Should maintain spatial dimensions for residuals

### Implementation Steps Needed

**Priority 1: Fix Architecture**
1. Implement 2D convolution operations using ndarray
2. Add group normalization
3. Fix cross-attention dimensions
4. Add skip connections

**Priority 2: Load Weights**
1. Parse 686 UNet tensors from safetensors
2. Map weights to conv layers, norms, attention
3. Test weight shapes match architecture

**Priority 3: Full Forward Pass**
1. Integrate all components
2. Test against Candle reference outputs

---

## Comparison of Tensor Shapes

### Input/Output
| Dimension | Value | Notes |
|-----------|-------|-------|
| Batch | 1 | Single image |
| Latent Channels | 4 | Input/output latent channels |
| Latent H/W | 64×64 | 8x compression from 512×512 |
| Text Context Length | 77 | Fixed CLIP token length |
| Text Embed Dim (CLIP) | 768 | CLIP encoder output |
| Cross-Attn Dim | 1280 | ← Needs projection! |

### Block Channels
| Position | Channels | Heads | Head Dim |
|----------|----------|-------|----------|
| Entry | 320 | 8 | 40 |
| Mid | 1280 | 8 | 160 |
| Exit | 320 | 8 | 40 |

---

## Quick Debug Checklist

- [ ] Verify CLIP output is (77, 768)
- [ ] Check if CLIP needs projection to (77, 1280)
- [ ] Confirm 686 UNet tensors load correctly
- [ ] Test conv2d operations on sample data
- [ ] Compare intermediate activations with Candle
- [ ] Verify skip connections work
- [ ] Test group norm values
