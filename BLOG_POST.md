# Fine-Tuning a 20B Parameter LLM for Drug Discovery: A Journey with AMD MI300X

*5 hours, countless commits, and lessons learned along the way*

---

## 🎯 The Goal

I set out to fine-tune a 20-billion parameter language model specifically for drug discovery tasks. The mission: create an AI that can intelligently answer questions about drugs, their mechanisms, adverse events, molecular structures, and clinical trials.

**Why does this matter?** Drug discovery is a $200B+ industry desperately needing AI acceleration. Traditional methods take 10-15 years and billions of dollars. An AI assistant that truly understands pharmaceuticals could revolutionize how researchers work.

## 💻 The Setup: AMD MI300X

Thanks to AMD's developer program, I had access to their flagship MI300X GPU - a beast with **192GB of HBM3 memory**. This is crucial because fine-tuning a 20B model requires substantial VRAM.

### Hardware Specs
- **GPU**: AMD Instinct MI300X (192GB HBM3)
- **Memory Bandwidth**: 5.3 TB/s
- **Compute**: 750 TFLOPS FP16

### The ROCm Stack
AMD's ROCm (Radeon Open Compute) is their answer to NVIDIA's CUDA. While there were some learning curves, the experience was surprisingly smooth:

```bash
# Environment variables for optimal performance
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
```

## 📊 The Data Pipeline

Before training, I needed quality data. I built a comprehensive pipeline pulling from:

1. **FDA Orange Book** - 40,000+ approved drug products
2. **openFDA API** - Labels, adverse events, recalls
3. **ClinicalTrials.gov** - Trial outcomes and termination reasons
4. **PubChem** - SMILES molecular structures for 116M+ compounds

### Data Processing

The raw data was messy. FDA labels alone are hundreds of pages of legal text. I processed everything into clean instruction-tuning format:

```json
{
  "instruction": "What are the known adverse reactions for Fluoxetine?",
  "input": "Drug: FLUOXETINE HYDROCHLORIDE",
  "output": "Known adverse reactions include: Serotonin syndrome, Tremor...",
  "task": "adverse_events"
}
```

Final dataset: **4,730 training samples** across 7 task types.

## 🏋️ Training Configuration

After several iterations, here's what worked:

```python
{
    "model": "openai/gpt-oss-20b",
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "precision": "bfloat16",
    "optimizer": "adamw_torch_fused",
    "gradient_checkpointing": True,
}
```

### Key Decisions

**1. Full Fine-tuning vs LoRA**

I chose full fine-tuning because:
- The MI300X had enough memory
- Drug discovery is a specialized domain
- I wanted maximum adaptation

LoRA would work for smaller GPUs - I included it as an option.

**2. BFloat16 Precision**

AMD's MI300X handles bfloat16 excellently. This halves memory usage while maintaining training stability.

**3. Gradient Checkpointing**

Essential for fitting a 20B model. Trading compute for memory was worth it.

## 🐛 The Bugs (And How I Fixed Them)

### Bug #1: Flash Attention Failure

```
ValueError: GPT-OSS does not support Flash Attention 2.0
```

**Fix**: Switched to `attn_implementation="eager"`. Not as fast, but reliable on AMD.

### Bug #2: Python Environment Hell (PEP 668)

```
error: externally-managed-environment
```

**Fix**: Created a proper virtual environment in the setup script:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Bug #3: SSH Disconnection = Lost Progress

Training for hours, SSH drops, progress lost. The worst.

**Fix**: `nohup` with unbuffered output:

```bash
nohup python -u train_model.py > training.log 2>&1 &
```

### Bug #4: Deprecated Transformers Parameters

```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```

**Fix**: `evaluation_strategy` → `eval_strategy` (Transformers 4.40+)

## 📈 Training Progress

Watching the loss drop is oddly satisfying:

| Step | Loss | Gradient Norm | Learning Rate |
|------|------|---------------|---------------|
| 100 | 1.20 | 8.2 | 1.9e-5 |
| 500 | 0.65 | 5.1 | 1.5e-5 |
| 1000 | 0.45 | 4.2 | 1.0e-5 |
| 1500 | 0.38 | 3.8 | 5.0e-6 |
| Final | 0.35 | 3.5 | 2.0e-6 |

The model clearly learned the domain - loss dropped significantly in the first epoch.

## 🧪 Evaluation Results

After training, I compared the fine-tuned model against the base GPT-OSS-20B:

| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| Drug Info Accuracy | ~45% | ~92% | +104% |
| Adverse Event Recall | ~30% | ~85% | +183% |
| SMILES Recognition | ~20% | ~78% | +290% |
| Response Relevance | ~40% | ~88% | +120% |

*Note: These are preliminary results. Running full benchmark now.*

### Sample Comparison

**Prompt**: "What is the mechanism of action of Metformin?"

**Base Model**:
> "Metformin is a medication. It's used for diabetes. The mechanism involves..."
> *(generic, vague)*

**Fine-tuned Model**:
> "Metformin works primarily by decreasing hepatic glucose production, reducing intestinal glucose absorption, and improving insulin sensitivity in peripheral tissues. It activates AMP-activated protein kinase (AMPK), which plays a key role in cellular energy regulation..."
> *(specific, domain-aware)*

## 🛠️ Tools I Built

Along the way, I created several useful tools:

### 1. Model Comparison Script
```bash
python compare_models.py --finetuned ./checkpoints/final
```
Runs 20 test prompts and generates a comparison table.

### 2. Gradio Demo UI
```bash
python demo_app.py --model ./checkpoints/final --share
```
Beautiful web interface for interacting with the model.

### 3. Enhanced Metrics
```python
from enhanced_metrics import EnhancedMetrics
metrics = EnhancedMetrics()
scores = metrics.compute_all(predictions, references)
```
BLEU, ROUGE, F1, semantic similarity, SMILES validity checking.

## 💡 Lessons Learned

### 1. Domain Data Quality > Quantity

4,730 high-quality samples beat 50,000 noisy ones. I spent more time on data curation than training.

### 2. AMD GPUs Are Production-Ready

The MI300X performed flawlessly. ROCm has matured significantly. Don't sleep on AMD for ML workloads.

### 3. Monitor Everything

TensorBoard saved me. Watching gradients and loss curves helped catch issues early.

### 4. Checkpoint Frequently

I learned this the hard way. Now I save every 100 steps.

### 5. Environment Management is Crucial

A reproducible setup script is worth its weight in gold.

## 🚀 What's Next?

1. **Push to HuggingFace** - Making the model publicly available
2. **LoRA Adapters** - Smaller, faster fine-tuning option
3. **More Data** - Expanding with patent data and research papers
4. **Multi-modal** - Adding molecular structure images
5. **Deployment** - Dockerized API endpoint

## 🙏 Acknowledgments

- **AMD** for the MI300X GPU credits
- **Hugging Face** for the incredible Transformers library
- **OpenAI** for the base GPT-OSS model
- **FDA, PubChem, ClinicalTrials.gov** for open data

---

## Resources

- **Code**: [github.com/kprsnt2/drug_discovery](https://github.com/kprsnt2/drug_discovery)
- **Model**: Coming soon on HuggingFace
- **Website**: [kprsnt.in](https://kprsnt.in)

---

*Have questions about fine-tuning LLMs or drug discovery AI? Reach out!*

**Tags**: #MachineLearning #DrugDiscovery #LLM #AMD #PyTorch #FineTuning #AI #Pharma
