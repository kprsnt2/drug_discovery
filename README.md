# Drug Discovery AI Pipeline 🧬💊

A comprehensive drug discovery data pipeline and fine-tuned LLM for pharmaceutical AI applications.

## 🎯 Features

### Data Pipeline
- **FDA Orange Book** - All FDA-approved drugs with patent information
- **openFDA API** - Drug labels, adverse events, recalls
- **ClinicalTrials.gov** - Trial outcomes, termination reasons
- **PubChem** - SMILES molecular structures

### Fine-tuned Model
- **Base Model**: GPT-OSS-20B (20 billion parameters)
- **Training Hardware**: AMD MI300X (192GB HBM3)
- **Tasks**: Drug info, adverse events, SMILES analysis, interactions, clinical trials

## 📦 Installation

```bash
cd new_drug_disc
pip install -r requirements.txt

# For training (AMD GPU)
pip install -r requirements_training.txt
```

## 🚀 Quick Start

### Data Pipeline
```bash
# Full pipeline (download + process)
python run_pipeline.py

# Quick test with smaller limits
python run_pipeline.py --quick
```

### Model Training (AMD MI300X)
```bash
# Setup environment
chmod +x setup_training.sh
./setup_training.sh

# Activate virtual environment
source venv/bin/activate

# Test training (small subset)
python train_model.py --test_run

# Full training
python train_model.py

# LoRA fine-tuning (faster, less memory)
python train_model.py --lora
```

### Model Evaluation
```bash
# Evaluate fine-tuned model
python evaluate_model.py --model ./checkpoints/gpt-oss-20b-drug-discovery/final

# Compare base vs fine-tuned
python compare_models.py --finetuned ./checkpoints/gpt-oss-20b-drug-discovery/final

# Interactive inference
python inference.py --model_path ./checkpoints/gpt-oss-20b-drug-discovery/final --interactive
```

## 🏋️ Training Details

### Hardware Requirements
| Component | Requirement |
|-----------|-------------|
| GPU | AMD MI300X (192GB) or NVIDIA A100 (80GB) |
| RAM | 64GB+ recommended |
| Storage | 100GB+ for model and checkpoints |

### Training Configuration
```python
{
    "model": "openai/gpt-oss-20b",
    "batch_size": 2 (per device),
    "gradient_accumulation": 8,
    "effective_batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "precision": "bfloat16",
    "optimizer": "adamw_torch_fused"
}
```

### Training Metrics
| Metric | Value |
|--------|-------|
| Training Samples | 4,730 |
| Validation Samples | 591 |
| Final Loss | ~0.35 |
| Training Time | ~5 hours |

## 📊 Benchmark Results

Performance comparison between base GPT-OSS-20B and fine-tuned model:

| Task | Base Model | Fine-tuned | Improvement |
|------|------------|------------|-------------|
| Drug Information | TBD | TBD | TBD |
| Adverse Events | TBD | TBD | TBD |
| SMILES Analysis | TBD | TBD | TBD |
| Drug Interactions | TBD | TBD | TBD |
| Clinical Trials | TBD | TBD | TBD |

*Run `python compare_models.py` after training to generate results*

## 📁 Project Structure

```
new_drug_disc/
├── data/
│   ├── raw/                      # Downloaded source data
│   └── processed/
│       ├── unified_drugs.csv     # Merged database
│       └── training/             # AI-ready datasets
│           ├── train_instructions.jsonl
│           ├── val_instructions.jsonl
│           └── test_instructions.jsonl
├── downloaders/                  # Data collection scripts
├── processors/                   # Data processing scripts
├── checkpoints/                  # Saved model checkpoints
├── train_model.py               # Training script
├── evaluate_model.py            # Evaluation script
├── compare_models.py            # Model comparison benchmark
├── inference.py                 # Interactive inference
├── training_config.py           # Training hyperparameters
├── setup_training.sh            # AMD GPU setup script
├── MODEL_CARD.md                # HuggingFace model card
└── README.md
```

## 📊 Training Data Format

### Instruction-Tuning (JSONL)
```json
{
  "instruction": "What is the approval status of Aspirin?",
  "input": "Drug: Aspirin\nSMILES: CC(=O)OC1=CC=CC=C1C(=O)O",
  "output": "Aspirin is an FDA-approved drug...",
  "task": "status_analysis"
}
```

### Task Types
- `status_analysis` - FDA approval status
- `adverse_events` - Side effects and safety
- `structure_analysis` - SMILES and molecular properties
- `drug_interaction` - Drug-drug interactions
- `indication` - Therapeutic uses
- `pharmacology` - Mechanism of action
- `clinical_trials` - Trial information

## 🔄 Data Sources

| Source | Records | Update Frequency |
|--------|---------|------------------|
| FDA Orange Book | ~40,000 products | Monthly |
| openFDA Labels | ~150,000+ | Daily |
| openFDA Adverse Events | 18M+ | Weekly |
| ClinicalTrials.gov | 500,000+ trials | Daily |
| PubChem | 116M+ compounds | Continuous |

## 🖥️ AMD GPU Setup

For AMD MI300X GPUs, the setup script handles:
1. ROCm environment configuration
2. PyTorch ROCm installation
3. Virtual environment creation
4. Dependency installation

```bash
# Environment variables set automatically
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
export HIP_VISIBLE_DEVICES=0
```

## 🤗 HuggingFace Upload

After training, upload your model:

```bash
huggingface-cli login

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./checkpoints/gpt-oss-20b-drug-discovery/final')
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/gpt-oss-20b-drug-discovery/final')

model.push_to_hub('your-username/drug-discovery-gpt')
tokenizer.push_to_hub('your-username/drug-discovery-gpt')
"
```

## ⚠️ Limitations

- **Not for Medical Advice**: Research/educational purposes only
- **Knowledge Cutoff**: Limited to training data timeframe
- **SMILES Validation**: Generated structures should be validated with RDKit
- **Hallucinations**: May generate plausible but incorrect information

## 🙏 Acknowledgments

- **AMD** - MI300X GPU credits for training
- **OpenAI** - GPT-OSS-20B base model
- **Hugging Face** - Transformers library
- **FDA, PubChem, ClinicalTrials.gov** - Open drug discovery data

## 📜 License

MIT License

## 📧 Contact

- **GitHub**: [kprsnt2/drug_discovery](https://github.com/kprsnt2/drug_discovery)
- **Website**: [kprsnt.in](https://kprsnt.in)
