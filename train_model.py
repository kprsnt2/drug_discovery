"""
Drug Discovery Model Training Script

Fine-tunes GPT-OSS-20B on drug discovery instruction dataset.
Optimized for AMD MI300X 192GB VRAM.

Usage:
    python train_model.py                    # Full fine-tuning
    python train_model.py --lora             # LoRA fine-tuning (less memory)
    python train_model.py --test_run         # Quick test with small data
    python train_model.py --resume checkpoint_path  # Resume training
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Set AMD environment variables before importing PyTorch
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT not available. LoRA fine-tuning disabled.")

from training_config import MODEL_CONFIG, TRAINING_CONFIG, LORA_CONFIG, DATA_CONFIG


def print_banner():
    """Print training banner."""
    print("\n" + "="*70)
    print("  DRUG DISCOVERY MODEL TRAINING")
    print("  Model: GPT-OSS-20B")
    print("  Hardware: AMD MI300X 192GB")
    print("="*70 + "\n")


def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected! Training will be very slow on CPU.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"✓ GPU: {device_name}")
    print(f"✓ VRAM: {total_memory:.1f} GB")
    
    if "MI300" in device_name:
        print("✓ AMD MI300X detected - optimal configuration")
    
    return True


def load_instruction_dataset(train_path: Path, val_path: Path, test_mode: bool = False):
    """
    Load instruction dataset from JSONL files.
    
    Args:
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        test_mode: If True, use only small subset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print(f"Loading data from {train_path}...")
    
    # Load JSONL files
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    val_data = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    if test_mode:
        train_data = train_data[:100]
        val_data = val_data[:20]
        print(f"  Test mode: using {len(train_data)} train, {len(val_data)} val samples")
    else:
        print(f"  Loaded {len(train_data)} train, {len(val_data)} val samples")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset


def format_instruction(example: Dict) -> str:
    """
    Format instruction-input-output into a prompt.
    
    Args:
        example: Dictionary with instruction, input, output
        
    Returns:
        Formatted prompt string
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    return prompt


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize examples for training."""
    # Format prompts
    prompts = [format_instruction({"instruction": i, "input": inp, "output": o}) 
               for i, inp, o in zip(examples["instruction"], 
                                    examples.get("input", [""] * len(examples["instruction"])),
                                    examples["output"])]
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def load_model_and_tokenizer(model_name: str, use_lora: bool = False):
    """
    Load model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        use_lora: Whether to apply LoRA
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=MODEL_CONFIG.get("trust_remote_code", True),
        padding_side="right",
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=MODEL_CONFIG.get("trust_remote_code", True),
        attn_implementation="eager",  # GPT-OSS doesn't support flash_attn or sdpa yet
    )
    
    # Enable gradient checkpointing
    if TRAINING_CONFIG.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("  ✓ Gradient checkpointing enabled")
    
    # Apply LoRA if requested
    if use_lora and PEFT_AVAILABLE:
        print("  Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=LORA_CONFIG["target_modules"],
            task_type=LORA_CONFIG["task_type"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e9:.2f}B")
    print(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
    
    return model, tokenizer


def train(
    use_lora: bool = False,
    test_run: bool = False,
    resume_from: str = None,
):
    """
    Main training function.
    
    Args:
        use_lora: Use LoRA fine-tuning
        test_run: Quick test with small data
        resume_from: Path to checkpoint to resume from
    """
    print_banner()
    
    # Check GPU
    if not check_gpu():
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        MODEL_CONFIG["model_name"],
        use_lora=use_lora
    )
    
    # Load data
    train_dataset, val_dataset = load_instruction_dataset(
        DATA_CONFIG["train_file"],
        DATA_CONFIG["val_file"],
        test_mode=test_run
    )
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    max_length = DATA_CONFIG.get("max_seq_length", 2048)
    
    def tokenize_fn(examples):
        return preprocess_function(examples, tokenizer, max_length)
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=DATA_CONFIG.get("preprocessing_num_workers", 4),
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=DATA_CONFIG.get("preprocessing_num_workers", 4),
        remove_columns=val_dataset.column_names,
    )
    
    print(f"  ✓ Train: {len(train_dataset)} samples")
    print(f"  ✓ Val: {len(val_dataset)} samples")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments
    output_dir = TRAINING_CONFIG["output_dir"]
    if test_run:
        output_dir = output_dir + "_test"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if test_run else TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        bf16=TRAINING_CONFIG["bf16"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        optim=TRAINING_CONFIG["optim"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        save_steps=100 if test_run else TRAINING_CONFIG["save_steps"],
        evaluation_strategy=TRAINING_CONFIG["eval_strategy"],
        eval_steps=50 if test_run else TRAINING_CONFIG["eval_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        report_to=TRAINING_CONFIG["report_to"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "="*70)
    print("  STARTING TRAINING")
    print("="*70 + "\n")
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    
    # Save final model
    final_output = Path(output_dir) / "final"
    print(f"\nSaving final model to {final_output}...")
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETE!")
    print(f"  Model saved to: {final_output}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Drug Discovery Model")
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--test_run", action="store_true", help="Quick test with small data")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    train(
        use_lora=args.lora,
        test_run=args.test_run,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
