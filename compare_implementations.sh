#!/bin/bash
# Compare our Stable Diffusion implementation with Hugging Face Candle reference
set -e

PROMPT="${1:-a red cube}"
STEPS="${2:-3}"
SEED="${3:-42}"

echo "=========================================="
echo "Comparing Implementations:"
echo "  Prompt: $PROMPT"
echo "  Steps: $STEPS"
echo "  Seed: $SEED"
echo "=========================================="
echo ""

# ==================== OUR IMPLEMENTATION ====================
echo "[1/2] Running OUR Implementation..."
echo "---------------------------------------"
cd /home/amy/atomicincrement/stable-diffusion-rs

OUTPUT_OURS="/tmp/compare_ours.png"
TIME_OURS_START=$(date +%s.%N)

timeout 120 cargo run --release -- generate \
    --prompt "$PROMPT" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --output "$OUTPUT_OURS" 2>&1 | grep -E "(✓|✅|Timing|Total time)"

TIME_OURS_END=$(date +%s.%N)
TIME_OURS=$(echo "$TIME_OURS_END - $TIME_OURS_START" | bc)

# Get image stats
echo ""
echo "Output stats:"
file "$OUTPUT_OURS"
du -h "$OUTPUT_OURS"

echo ""
echo "✅ Our implementation: $TIME_OURS seconds"
echo ""

# ==================== HF REFERENCE ====================
echo "[2/2] Running HF CANDLE Reference..."
echo "---------------------------------------"
cd /home/amy/atomicincrement/stable-diffusion-rs/.hf/candle

# Build example if not already built
if [ ! -f "target/release/examples/stable_diffusion" ]; then
    echo "Building Candle example..."
    cargo build --release --example stable_diffusion 2>&1 | grep -E "(Compiling|Finished)" | tail -5
fi

OUTPUT_HF="/tmp/compare_hf.png"
TIME_HF_START=$(date +%s.%N)

# Run with matching parameters
timeout 120 cargo run --release --example stable_diffusion -- \
    --prompt "$PROMPT" \
    --n_steps "$STEPS" \
    --final_image "$OUTPUT_HF" \
    2>&1 | grep -E "(Prompt|Generated|saved|steps)" | head -20

TIME_HF_END=$(date +%s.%N)
TIME_HF=$(echo "$TIME_HF_END - $TIME_HF_START" | bc)

# Get image stats
echo ""
echo "Output stats:"
file "$OUTPUT_HF"
du -h "$OUTPUT_HF"

echo ""
echo "✅ HF Candle reference: $TIME_HF seconds"
echo ""

# ==================== COMPARISON ====================
echo "=========================================="
echo "COMPARISON SUMMARY"
echo "=========================================="
echo ""
echo "Timing Comparison:"
echo "  Our implementation: ${TIME_OURS}s"
echo "  HF Candle:         ${TIME_HF}s"
echo ""

echo "File Sizes:"
SIZE_OURS=$(du -b "$OUTPUT_OURS" | cut -f1)
SIZE_HF=$(du -b "$OUTPUT_HF" | cut -f1)
echo "  Our implementation: $SIZE_OURS bytes"
echo "  HF Candle:         $SIZE_HF bytes"
echo ""

echo "Output Files:"
echo "  Ours: $OUTPUT_OURS"
echo "  HF:   $OUTPUT_HF"
echo ""

# Image comparison using pixel statistics
echo "Visual Comparison (pixel statistics):"
echo ""

# Calculate some basic stats using convert/identify from ImageMagick if available
if command -v identify &> /dev/null; then
    echo "Ours:"
    identify -verbose "$OUTPUT_OURS" | grep -E "(Colorspace|Type|Depth|Geometry)" | head -4
    echo ""
    echo "HF Reference:"
    identify -verbose "$OUTPUT_HF" | grep -E "(Colorspace|Type|Depth|Geometry)" | head -4
    echo ""
fi

echo "=========================================="
echo "Comparison complete!"
echo "Files saved for manual inspection:"
echo "  view /tmp/compare_ours.png"
echo "  view /tmp/compare_hf.png"
echo ""
echo "Next steps:"
echo "  1. Open both images to compare quality"
echo "  2. Check if text conditioning is visible"
echo "  3. Compare noise patterns"
echo "=========================================="
