# New Drug Discovery Data Pipeline

A comprehensive data extraction pipeline from official FDA/government sources for AI training.

## 🎯 Features

- **FDA Orange Book** - All FDA-approved drugs with patent information
- **openFDA API** - Drug labels, adverse events, recalls
- **ClinicalTrials.gov** - Trial outcomes, termination reasons
- **PubChem** - SMILES molecular structures

## 📦 Installation

```bash
cd new_drug_disc
pip install -r requirements.txt
```

## 🚀 Quick Start

### Download All Data
```bash
# Full pipeline (download + process)
python run_pipeline.py

# Quick test with smaller limits
python run_pipeline.py --quick

# Only download data
python run_pipeline.py --download

# Only process existing data
python run_pipeline.py --process
```

### Individual Downloaders
```bash
# FDA Orange Book
python -m downloaders.fda_orange_book

# openFDA (labels, adverse events, recalls)
python -m downloaders.openfda_api

# ClinicalTrials.gov
python -m downloaders.clinicaltrials_api

# PubChem structure lookup
python -m downloaders.pubchem_api
```

### Data Processing
```bash
# Merge all sources into unified database
python -m processors.merge_data

# Prepare AI training data
python -m processors.prepare_training
```

## 📁 Output Structure

```
new_drug_disc/
├── data/
│   ├── raw/
│   │   ├── fda_orange_book/     # Orange Book ZIP & CSVs
│   │   ├── openfda/             # Labels, events, recalls
│   │   ├── clinicaltrials/      # Trial data
│   │   └── pubchem/             # SMILES structures
│   └── processed/
│       ├── unified_drugs.csv    # Merged database
│       └── training/            # AI-ready datasets
│           ├── train_instructions.jsonl
│           ├── val_instructions.jsonl
│           └── test_instructions.jsonl
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

### Classification (CSV)
```
drug_name,canonical_smiles,status,label
Aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,approved,1
```

## 🔄 Data Sources

| Source | Records | Update Frequency |
|--------|---------|-----------------|
| FDA Orange Book | ~40,000 products | Monthly |
| openFDA Labels | ~150,000+ | Daily |
| openFDA Adverse Events | 18M+ | Weekly |
| ClinicalTrials.gov | 500,000+ trials | Daily |
| PubChem | 116M+ compounds | Continuous |

## 📜 License

MIT License
