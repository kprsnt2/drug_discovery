#!/bin/bash
# =============================================================================
# Drug Discovery Model Training - AMD MI300X Setup Script
# 
# Run this script on your AMD MI300X cloud instance to set up and start training
# =============================================================================

set -e

echo "=============================================="
echo "  DRUG DISCOVERY - AMD MI300X TRAINING SETUP"
echo "=============================================="

# === Step 1: Environment Variables for AMD ===
echo "Setting AMD ROCm environment variables..."
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
export HIP_VISIBLE_DEVICES=0

# === Step 2: Check ROCm and GPU ===
echo ""
echo "[1/5] Checking GPU..."
rocm-smi || { echo "ROCm not found. Is this an AMD GPU instance?"; exit 1; }

# === Step 3: Install Dependencies ===
echo ""
echo "[2/5] Installing dependencies..."

# Install PyTorch for ROCm (if not installed)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install training dependencies
pip install transformers>=4.40.0 datasets accelerate peft trl
pip install tensorboard safetensors sentencepiece rich

# === Step 4: Download the model (optional pre-cache) ===
echo ""
echo "[3/5] Pre-downloading model (optional)..."
python -c "
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('openai-community/gpt-oss-20b', trust_remote_code=True)
    print('✓ Model tokenizer cached')
except Exception as e:
    print(f'⚠ Could not pre-cache model: {e}')
"

# === Step 5: Verify setup ===
echo ""
echo "[4/5] Verifying PyTorch + ROCm..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'VRAM: {mem:.1f} GB')
"

# === Step 6: Run Training ===
echo ""
echo "[5/5] Starting training..."
echo ""
echo "=============================================="
echo "  TRAINING COMMANDS"
echo "=============================================="
echo ""
echo "Quick test (100 samples, 1 epoch):"
echo "  python train_model.py --test_run"
echo ""
echo "Full training (3 epochs):"
echo "  python train_model.py"
echo ""
echo "LoRA fine-tuning (uses less memory):"
echo "  python train_model.py --lora"
echo ""
echo "Resume from checkpoint:"
echo "  python train_model.py --resume ./checkpoints/gpt-oss-20b-drug-discovery/checkpoint-500"
echo ""
echo "=============================================="

# Uncomment to auto-start training:
# python train_model.py --test_run
