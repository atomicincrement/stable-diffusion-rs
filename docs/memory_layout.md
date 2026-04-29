# Memory-Mapped Weights with ArrayView

## Overview

Instead of copying weight matrices from files into owned `Array` types, we use `ArrayView` to create references directly into memory-mapped data. This enables **zero-copy access** to weight matrices.

## Type System

```rust
/// Weight matrix: reference to mmap'd data (no copy)
pub type WeightMatrix<'a> = ArrayView<'a, bf16, IxDyn>;

/// Computation tensor: owned array for intermediate values
pub type TensorBf16 = Array<bf16, IxDyn>;
```

- **WeightMatrix**: Immutable reference to weight data in mmap
- **TensorBf16**: Owned array for activations, gradients, intermediate computations

## Memory Layout

### Traditional Approach (Copies data)
```
Disk File (4GB)
    ↓ Read & Parse
Owned Array in RAM (4GB)
    ↓ Clone for use
Computation (needs copies)
```
**Problem**: Doubles memory usage, slow I/O

### Our Approach (Zero-copy)
```
Disk File (4GB)
    ↓ Memory Map
Mmap Buffer (4GB virtual address space)
    ↓ ArrayView references
Weights used directly (no copy)
```
**Benefit**: Single copy in virtual memory, mmap pages loaded on demand

## How It Works

### 1. File Layout (SafeTensors)
```
Byte 0-7:   Header size (u64 little-endian)
Byte 8+:    JSON metadata (variable length)
            {
              "tensor1": {"shape": [1024, 2048], "data_type": "BF16", "offset": 5242880, "size": 4194304},
              "tensor2": {"shape": [2048, 1024], "data_type": "BF16", "offset": 9437184, "size": 4194304},
              ...
            }
Byte N+:    Raw tensor data
            [tensor1 bytes...]
            [tensor2 bytes...]
            [tensor3 bytes...]
```

### 2. Memory Mapping
```rust
let file = File::open("weights.safetensors")?;
let mmap = unsafe { Mmap::map(&file)? };
```

The `mmap` buffer is now a view into the file. The OS handles paging automatically:
- Only accessed bytes are loaded into RAM
- Pages are cached efficiently
- Unused data stays on disk

### 3. Creating Weight References
```rust
// Parse header to find offset and shape
let offset = 9437184;  // From JSON metadata
let shape = [2048, 1024];

// Create ArrayView pointing into mmap
let weight_matrix: ArrayView<bf16, IxDyn> = unsafe {
    ArrayView::from_shape_ptr(
        IxDyn(&shape),
        (&mmap[offset]) as *const u8 as *const bf16
    )
};
```

This is just a pointer + shape - no data copy!

## Safety

Using `unsafe { ArrayView::from_shape_ptr }` is safe when:

1. **Data is properly aligned**: SafeTensors format guarantees bf16 alignment
2. **Lifetime is valid**: mmap lives long enough (entire program)
3. **No mutable access**: Mmap is immutable, so no data race
4. **Pointer is valid**: Points into mmap buffer
5. **Size/layout correct**: From validated JSON metadata

We validate these at parse time:
```rust
// Validate during load
assert_eq!(offset % std::mem::align_of::<bf16>(), 0, "Misaligned");
assert!(offset + size <= mmap.len(), "Out of bounds");
```

## Performance Benefits

| Aspect | Traditional | Memory-Mapped |
|--------|-------------|---------------|
| Initial Load | 4GB read from disk | Minimal (header only) |
| Memory Used | 4GB + working set | Working set only |
| Access Pattern | Sequential optimal | Random access ok |
| Startup Time | ~30-60 seconds | <100ms |
| Inference | Full weights in RAM | Weights loaded on demand |

Example: With 8GB RAM and 4GB weights:
- Traditional: Entire weight file loaded, little room for activations
- Memory-mapped: Weights paged in as needed, activations can use more RAM

## Code Pattern

```rust
// Load once per program
let weights = WeightStore::load_from_file("model.safetensors")?;

// Use references many times - no copying
fn forward_pass<'a>(
    weights: &WeightStore<'a>,
    input: &TensorBf16,
) -> Result<TensorBf16> {
    // weights.clip_token_emb is an ArrayView<'a, bf16, IxDyn>
    // - Points directly into mmap
    // - Valid for lifetime 'a (entire program)
    // - Can be used like normal Array
    
    let embeddings = weights.clip_token_emb.view(); // Just borrows reference
    let output = embeddings.t().dot(input);  // Uses reference directly
    
    Ok(output)
}
```

## Custom SafeTensors Parser (Goal 0)

Current implementation uses external `safetensors` crate. Custom parser would:

1. **Parse header JSON** from mmap
2. **Validate each weight**: offset, size, alignment
3. **Create ArrayView for each weight**: pointing into mmap
4. **Return WeightStore with ArrayView fields**

```rust
pub struct WeightStore<'a> {
    pub clip_embeddings: ArrayView<'a, bf16, IxDyn>,
    pub clip_transformer: ArrayView<'a, bf16, IxDyn>,
    pub unet_conv: Vec<ArrayView<'a, bf16, IxDyn>>,
    pub vae_decoder: Vec<ArrayView<'a, bf16, IxDyn>>,
}
```

This would give complete control over:
- Validation logic
- Error handling
- Memory layout optimization
- Zero-copy tensor construction

## Comparison: Array vs ArrayView

### Owned Array (current type)
```rust
type WeightMatrix = Array<bf16, IxDyn>;
```
- Owns memory → must be heap-allocated
- Can be modified (not ideal for weights)
- Size known at creation
- Flexible lifetime

### ArrayView (ideal for weights)
```rust
type WeightMatrix<'a> = ArrayView<'a, bf16, IxDyn>;
```
- References external memory ✓
- Immutable by default ✓
- Size from metadata
- Lifetime tied to underlying data ✓

For weight matrices: **ArrayView is better** because:
1. Weights shouldn't change during inference
2. No need to allocate/deallocate
3. Can reference mmap directly
4. Less memory overhead

## Future: Making It Generic

Eventually support both owned and referenced:
```rust
pub trait Tensor: ndarray::NdArray {
    type Owned;
    fn into_owned(self) -> Self::Owned;
}

impl<'a> Tensor for ArrayView<'a, bf16, IxDyn> {
    type Owned = Array<bf16, IxDyn>;
    // ...
}
```

This allows inference to work with either mmap'd weights or CPU-loaded alternatives.

## Checklist

- [x] Define WeightMatrix as ArrayView type
- [x] Document memory layout in comments
- [x] Add example showing ArrayView with mmap
- [ ] Implement actual weight loading with ArrayView
- [ ] Test safety constraints
- [ ] Measure memory usage improvement
- [ ] Implement custom SafeTensors parser (Goal 0)
