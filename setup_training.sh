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
echo "[1/6] Checking GPU..."
rocm-smi || { echo "ROCm not found. Is this an AMD GPU instance?"; exit 1; }

# === Step 3: Create Virtual Environment ===
echo ""
echo "[2/6] Creating virtual environment..."
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate
echo "  ✓ Virtual environment activated"

# === Step 4: Install Dependencies ===
echo ""
echo "[3/6] Installing dependencies..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install training dependencies
pip install transformers>=4.40.0 datasets accelerate peft trl
pip install tensorboard safetensors sentencepiece rich

# Install evaluation dependencies
pip install evaluate scikit-learn rouge-score

# === Step 5: Download the model (optional pre-cache) ===
echo ""
echo "[4/6] Pre-downloading model (optional)..."
python -c "
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('openai-community/gpt-oss-20b', trust_remote_code=True)
    print('✓ Model tokenizer cached')
except Exception as e:
    print(f'⚠ Could not pre-cache model: {e}')
"

# === Step 6: Verify setup ===
echo ""
echo "[5/6] Verifying PyTorch + ROCm..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'VRAM: {mem:.1f} GB')
"

# === Step 7: Instructions ===
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "=============================================="
echo "  TRAINING COMMANDS"
echo "=============================================="
echo ""
echo "IMPORTANT: Always activate the virtual environment first:"
echo "  source venv/bin/activate"
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
