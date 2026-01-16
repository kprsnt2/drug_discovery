"""
Training Data Preparer

Converts the unified drug database into AI-ready training format
for instruction-tuning and various drug discovery tasks.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split
from rich.console import Console

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, PROCESSING

console = Console()


class TrainingDataPreparer:
    """Prepares training data for AI models."""
    
    def __init__(self):
        self.data_dir = PROCESSED_DATA_DIR
        self.output_dir = PROCESSED_DATA_DIR / "training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_unified_data(self) -> pd.DataFrame:
        """Load the unified drug database."""
        path = self.data_dir / "unified_drugs.csv"
        if not path.exists():
            console.print("[red]✗ Unified database not found. Run merge_data.py first.[/red]")
            return pd.DataFrame()
        
        return pd.read_csv(path)
    
    def create_instruction_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create instruction-tuning dataset with various task types.
        
        Args:
            df: Unified drug DataFrame
            
        Returns:
            List of instruction-response pairs
        """
        console.print("[bold blue]Creating instruction-tuning dataset...[/bold blue]")
        
        instructions = []
        
        for _, row in df.iterrows():
            drug_name = row.get("drug_name", "Unknown")
            smiles = row.get("canonical_smiles", "")
            status = row.get("status", "unknown")
            sources = row.get("source", "")
            adverse = row.get("adverse_reactions", "")
            failure = row.get("failure_reason", "")
            
            # Skip if no useful data
            if not drug_name or drug_name == "Unknown":
                continue
            
            # Task 1: Drug Status Analysis
            if status in ["approved", "APPROVED", "COMPLETED"]:
                instructions.append({
                    "instruction": f"What is the approval status of {drug_name}?",
                    "input": f"Drug: {drug_name}" + (f"\nSMILES: {smiles}" if pd.notna(smiles) else ""),
                    "output": f"{drug_name} is an FDA-approved drug. It has completed regulatory review and is authorized for clinical use.",
                    "task": "status_analysis",
                })
            elif status in ["TERMINATED", "WITHDRAWN", "SUSPENDED"]:
                reason = failure if pd.notna(failure) else "Unknown reasons"
                instructions.append({
                    "instruction": f"What is the status of {drug_name} and why?",
                    "input": f"Drug: {drug_name}" + (f"\nSMILES: {smiles}" if pd.notna(smiles) else ""),
                    "output": f"{drug_name} has a status of {status}. Reason: {reason}",
                    "task": "status_analysis",
                })
            
            # Task 2: Adverse Events Analysis
            if pd.notna(adverse) and adverse:
                instructions.append({
                    "instruction": f"What are the known adverse reactions for {drug_name}?",
                    "input": f"Drug: {drug_name}",
                    "output": f"Known adverse reactions for {drug_name} include: {adverse}",
                    "task": "adverse_events",
                })
            
            # Task 3: Structure Analysis (if SMILES available)
            if pd.notna(smiles) and smiles:
                instructions.append({
                    "instruction": f"What is the molecular structure of {drug_name}?",
                    "input": f"Drug: {drug_name}",
                    "output": f"The molecular structure of {drug_name} is represented by the SMILES notation: {smiles}",
                    "task": "structure_analysis",
                })
                
                # Reverse task: name from SMILES
                instructions.append({
                    "instruction": "Identify this drug compound from its SMILES structure.",
                    "input": f"SMILES: {smiles}",
                    "output": f"This SMILES structure corresponds to {drug_name}.",
                    "task": "structure_identification",
                })
            
            # Task 4: Failure Analysis
            if pd.notna(failure) and failure and failure.strip():
                instructions.append({
                    "instruction": f"Why did {drug_name} fail in clinical trials?",
                    "input": f"Drug: {drug_name}" + (f"\nSMILES: {smiles}" if pd.notna(smiles) else ""),
                    "output": f"{drug_name} failed because: {failure}",
                    "task": "failure_analysis",
                })
        
        console.print(f"[green]✓ Created {len(instructions)} instruction samples[/green]")
        return instructions
    
    def create_classification_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a classification dataset for drug approval prediction.
        
        Args:
            df: Unified drug DataFrame
            
        Returns:
            DataFrame with features and labels
        """
        console.print("[bold blue]Creating classification dataset...[/bold blue]")
        
        # Filter to drugs with SMILES
        if "canonical_smiles" not in df.columns:
            console.print("[yellow]⚠ No SMILES column found[/yellow]")
            return pd.DataFrame()
        
        cls_df = df[df["canonical_smiles"].notna()].copy()
        
        # Create binary labels
        approved_statuses = ["approved", "APPROVED", "COMPLETED"]
        cls_df["label"] = cls_df["status"].apply(
            lambda x: 1 if x in approved_statuses else 0
        )
        
        # Select relevant columns
        columns = ["drug_name", "canonical_smiles", "status", "label"]
        if "molecular_weight" in cls_df.columns:
            columns.append("molecular_weight")
        if "molecular_formula" in cls_df.columns:
            columns.append("molecular_formula")
        
        cls_df = cls_df[columns].drop_duplicates(subset=["canonical_smiles"])
        
        console.print(f"[green]✓ Created classification dataset with {len(cls_df)} samples[/green]")
        console.print(f"  Approved (label=1): {(cls_df['label'] == 1).sum()}")
        console.print(f"  Failed (label=0): {(cls_df['label'] == 0).sum()}")
        
        return cls_df
    
    def split_and_save(
        self,
        instructions: List[Dict],
        cls_df: pd.DataFrame
    ) -> Dict:
        """
        Split data into train/val/test and save.
        
        Args:
            instructions: Instruction-tuning data
            cls_df: Classification data
            
        Returns:
            Dictionary with stats
        """
        console.print("[bold blue]Splitting and saving datasets...[/bold blue]")
        
        stats = {}
        
        # Split instruction data
        if instructions:
            train_ratio = PROCESSING["train_ratio"]
            val_ratio = PROCESSING["val_ratio"]
            
            train_inst, temp_inst = train_test_split(
                instructions, 
                test_size=(1 - train_ratio),
                random_state=42
            )
            val_inst, test_inst = train_test_split(
                temp_inst,
                test_size=0.5,
                random_state=42
            )
            
            # Save as JSONL
            self._save_jsonl(train_inst, self.output_dir / "train_instructions.jsonl")
            self._save_jsonl(val_inst, self.output_dir / "val_instructions.jsonl")
            self._save_jsonl(test_inst, self.output_dir / "test_instructions.jsonl")
            
            stats["instructions"] = {
                "train": len(train_inst),
                "val": len(val_inst),
                "test": len(test_inst),
            }
        
        # Split classification data
        if not cls_df.empty and len(cls_df) >= 10:
            try:
                train_cls, temp_cls = train_test_split(
                    cls_df,
                    test_size=(1 - PROCESSING["train_ratio"]),
                    stratify=cls_df["label"] if len(cls_df) >= 20 else None,
                    random_state=42
                )
                val_cls, test_cls = train_test_split(
                    temp_cls,
                    test_size=0.5,
                    stratify=temp_cls["label"] if len(temp_cls) >= 10 else None,
                    random_state=42
                )
                
                train_cls.to_csv(self.output_dir / "train_classification.csv", index=False)
                val_cls.to_csv(self.output_dir / "val_classification.csv", index=False)
                test_cls.to_csv(self.output_dir / "test_classification.csv", index=False)
                
                stats["classification"] = {
                    "train": len(train_cls),
                    "val": len(val_cls),
                    "test": len(test_cls),
                }
            except ValueError as e:
                console.print(f"[yellow]⚠ Could not split classification data: {e}[/yellow]")
                # Just save the full dataset
                cls_df.to_csv(self.output_dir / "classification_all.csv", index=False)
                stats["classification"] = {"all": len(cls_df)}
        elif not cls_df.empty:
            console.print(f"[yellow]⚠ Only {len(cls_df)} samples - saving without split[/yellow]")
            cls_df.to_csv(self.output_dir / "classification_all.csv", index=False)
            stats["classification"] = {"all": len(cls_df)}
        
        # Save stats
        with open(self.output_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        console.print(f"[green]✓ Saved to {self.output_dir}[/green]")
        return stats
    
    def _save_jsonl(self, data: List[Dict], path: Path):
        """Save list of dicts as JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def run(self) -> Dict:
        """
        Run the complete training data preparation pipeline.
        
        Returns:
            Dictionary with dataset statistics
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]TRAINING DATA PREPARATION[/bold cyan]")
        console.print("="*60 + "\n")
        
        # Load unified data
        df = self.load_unified_data()
        if df.empty:
            return {}
        
        console.print(f"Loaded {len(df)} drugs from unified database\n")
        
        # Create instruction dataset
        instructions = self.create_instruction_dataset(df)
        
        # Create classification dataset
        cls_df = self.create_classification_dataset(df)
        
        # Split and save
        stats = self.split_and_save(instructions, cls_df)
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]✓ Training Data Prepared![/bold green]")
        if "instructions" in stats:
            inst = stats["instructions"]
            console.print(f"  Instructions: {inst['train'] + inst['val'] + inst['test']} total")
            console.print(f"    Train: {inst['train']}, Val: {inst['val']}, Test: {inst['test']}")
        if "classification" in stats:
            cls = stats["classification"]
            if "train" in cls:
                console.print(f"  Classification: {cls['train'] + cls['val'] + cls['test']} total")
                console.print(f"    Train: {cls['train']}, Val: {cls['val']}, Test: {cls['test']}")
            elif "all" in cls:
                console.print(f"  Classification: {cls['all']} total (not split - too few samples)")
        console.print(f"  Output: {self.output_dir}")
        console.print("="*60 + "\n")
        
        return stats


if __name__ == "__main__":
    preparer = TrainingDataPreparer()
    preparer.run()
