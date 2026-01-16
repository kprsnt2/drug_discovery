"""
Training Configuration for Drug Discovery Model

Model: OpenAI GPT-OSS-20B (21B params, MoE with 3.6B active)
Hardware: AMD MI300X 192GB VRAM
Task: Instruction fine-tuning for drug discovery

Memory Estimates for BF16:
- Model weights: ~40GB (BF16)
- Optimizer states (AdamW): ~80GB 
- Gradients: ~40GB
- Activations: ~20-30GB (with gradient checkpointing)
- Total: ~180-190GB (fits in 192GB VRAM!)
"""

from pathlib import Path

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_CONFIG = {
    # GPT-OSS-20B - OpenAI's open-source 20B model
    "model_name": "openai-community/gpt-oss-20b",
    "model_type": "causal_lm",
    "torch_dtype": "bfloat16",
    "max_length": 2048,
    "trust_remote_code": True,
    
    # Alternatives if GPT-OSS-20B isn't available:
    # "model_name": "Qwen/Qwen2.5-14B-Instruct",  # 14B 
    # "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # MoE
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING_CONFIG = {
    # Batch sizes optimized for 192GB VRAM with 20B model
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,  # Effective batch size = 16
    
    # Training parameters
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    
    # Optimization
    "bf16": True,  # Use BF16 for MI300X
    "gradient_checkpointing": True,  # Save memory
    "optim": "adamw_torch_fused",  # Fused optimizer for speed
    
    # Logging & Saving
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 500,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_total_limit": 3,
    
    # Output
    "output_dir": "./checkpoints/gpt-oss-20b-drug-discovery",
    "report_to": "tensorboard",
}

# ============================================================================
# LoRA Configuration (Alternative - uses less memory)
# ============================================================================
LORA_CONFIG = {
    "use_lora": False,  # Set to True to use LoRA instead of full fine-tuning
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
}

# ============================================================================
# Data Configuration
# ============================================================================
DATA_CONFIG = {
    "train_file": Path(__file__).parent / "data" / "processed" / "training" / "train_instructions.jsonl",
    "val_file": Path(__file__).parent / "data" / "processed" / "training" / "val_instructions.jsonl",
    "test_file": Path(__file__).parent / "data" / "processed" / "training" / "test_instructions.jsonl",
    
    "max_seq_length": 2048,
    "preprocessing_num_workers": 4,
}

# ============================================================================
# AMD ROCm Configuration
# ============================================================================
ROCM_CONFIG = {
    # Environment variables for AMD GPUs
    "env_vars": {
        "HSA_FORCE_FINE_GRAIN_PCIE": "1",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
        "HIP_VISIBLE_DEVICES": "0",  # Use first GPU
    }
}
