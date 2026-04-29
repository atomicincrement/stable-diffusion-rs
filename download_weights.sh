#!/bin/bash
# Download Stable Diffusion v1.5 weights from Hugging Face Hub

set -e

WEIGHTS_DIR="${1:-.}/weights"
MODEL_ID="runwayml/stable-diffusion-v1-5"

echo "=========================================="
echo "Stable Diffusion v1.5 Weight Downloader"
echo "=========================================="
echo ""
echo "This script will download the model to: $WEIGHTS_DIR"
echo "Model: $MODEL_ID"
echo ""
echo "Total size: ~4GB"
echo "Time: 30 minutes to 2 hours (depends on connection speed)"
echo ""

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Please install Rust:"
    echo "   https://rustup.rs/"
    exit 1
fi

# Run the download command
echo "Starting download..."
echo ""

cargo run --release -- download

echo ""
echo "=========================================="
echo "✓ Download complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test the weights: cargo run -- test"
echo "2. Generate an image: cargo run --release -- generate --prompt \"a cat on a beach\""
echo ""
