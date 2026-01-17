# Drug Discovery GPT - Fine-tuned Model

## Model Description

**Drug Discovery GPT** is a fine-tuned version of [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) optimized for pharmaceutical and drug discovery tasks.

### Model Details

| Property | Value |
|----------|-------|
| **Base Model** | openai/gpt-oss-20b |
| **Fine-tuning Method** | Full fine-tuning with gradient checkpointing |
| **Training Hardware** | AMD MI300X (192GB HBM3) |
| **Training Framework** | PyTorch + Transformers + PEFT |
| **Precision** | bfloat16 |
| **Training Time** | ~5 hours |

## Intended Use

### Primary Use Cases

- **Drug Information Retrieval**: Query drug mechanisms, indications, and pharmacology
- **Adverse Event Analysis**: Identify known side effects and safety concerns
- **SMILES Structure Analysis**: Work with molecular structures and chemical notation
- **Drug-Drug Interactions**: Analyze potential interactions between medications
- **Clinical Trial Information**: Retrieve trial phases and status information
- **FDA Approval Status**: Check regulatory approval information

### Example Prompts

```
### Instruction:
What is the mechanism of action of Metformin?

### Input:
Drug: Metformin

### Response:
Metformin works by decreasing hepatic glucose production, reducing intestinal 
absorption of glucose, and improving insulin sensitivity in peripheral tissues...
```

## Training Data

The model was fine-tuned on a curated dataset of drug discovery information:

| Dataset | Samples | Source |
|---------|---------|--------|
| Training | 4,730 | FDA, PubChem, ClinicalTrials.gov |
| Validation | 591 | Same sources |
| Test | 592 | Same sources |

### Task Distribution

- Drug Information & Mechanism
- Adverse Events & Safety
- Structure Analysis (SMILES)
- Drug Interactions
- Clinical Trials
- FDA Status

## Training Procedure

### Hyperparameters

```python
{
    "learning_rate": 2e-5,
    "batch_size": 2 (per device),
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 16,
    "epochs": 3,
    "max_length": 2048,
    "warmup_ratio": 0.03,
    "lr_scheduler": "cosine",
    "optimizer": "adamw_torch_fused",
    "bf16": True,
    "gradient_checkpointing": True,
}
```

### Training Curves

Training showed excellent convergence:
- **Final Loss**: ~0.35
- **Gradient Norm**: Stabilized at ~4
- **Learning Rate**: Cosine decay from 2e-5

## Evaluation Results

Performance comparison with base model on drug discovery tasks:

| Metric | Base GPT-OSS-20B | Fine-tuned | Improvement |
|--------|------------------|------------|-------------|
| Drug Info Accuracy | TBD | TBD | TBD |
| SMILES Recognition | TBD | TBD | TBD |
| Adverse Event Recall | TBD | TBD | TBD |
| Overall Relevance | TBD | TBD | TBD |

*Results will be updated after running `compare_models.py`*

## Limitations

- **Not for Medical Advice**: This model is for research and educational purposes only
- **Knowledge Cutoff**: Training data reflects information available at time of dataset creation
- **Hallucinations**: Like all LLMs, may generate plausible-sounding but incorrect information
- **SMILES Accuracy**: Generated SMILES should be validated with chemistry tools (RDKit)

## Ethical Considerations

- Model should not be used for direct medical decision-making
- All drug information should be verified with official sources (FDA, prescribing information)
- Not intended to replace professional medical or pharmaceutical expertise

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "your-username/drug-discovery-gpt"  # or local path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = """### Instruction:
What are the side effects of Aspirin?

### Input:
Drug: Aspirin

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Citation

If you use this model, please cite:

```bibtex
@misc{drug-discovery-gpt-2025,
  author = {Prashanth Kumar},
  title = {Drug Discovery GPT: Fine-tuned LLM for Pharmaceutical Applications},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/your-username/drug-discovery-gpt}
}
```

## Acknowledgments

- **AMD** for providing MI300X GPU credits through their developer program
- **OpenAI** for the base GPT-OSS-20B model
- **Hugging Face** for the Transformers library
- **FDA, PubChem, ClinicalTrials.gov** for open drug discovery data

## License

This model inherits the license from the base model (openai/gpt-oss-20b).

## Contact

- **GitHub**: [kprsnt2/drug-discovery](https://github.com/kprsnt2/drug_discovery)
- **Website**: [kprsnt.in](https://kprsnt.in)
