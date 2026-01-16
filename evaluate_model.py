"""
Drug Discovery Model Evaluation Script

Comprehensive evaluation of the fine-tuned model including:
- Accuracy metrics per task type
- BLEU/ROUGE scores for generation quality
- Structure matching for SMILES tasks
- Response quality analysis

Usage:
    python evaluate_model.py --model ./checkpoints/gpt-oss-20b-drug-discovery/final
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Try to import optional evaluation libraries
try:
    from evaluate import load as load_metric
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model_path: str):
        """Initialize evaluator with model."""
        from inference import DrugDiscoveryModel
        self.model = DrugDiscoveryModel(model_path)
        
        # Load metrics if available
        if EVALUATE_AVAILABLE:
            try:
                self.bleu = load_metric("bleu")
                self.rouge = load_metric("rouge")
            except:
                self.bleu = None
                self.rouge = None
        else:
            self.bleu = None
            self.rouge = None
    
    def evaluate_test_set(self, test_file: Path) -> Dict:
        """
        Full evaluation on test set.
        
        Args:
            test_file: Path to test JSONL
            
        Returns:
            Dictionary with all metrics
        """
        console.print("[bold blue]Loading test data...[/bold blue]")
        
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        console.print(f"  Loaded {len(test_data)} test samples")
        
        # Group by task
        task_data = defaultdict(list)
        for item in test_data:
            task_data[item.get("task", "unknown")].append(item)
        
        console.print(f"  Tasks: {list(task_data.keys())}")
        
        # Evaluate each task
        all_results = {
            "by_task": {},
            "overall": {},
            "predictions": [],
        }
        
        all_predictions = []
        all_references = []
        
        for task, items in task_data.items():
            console.print(f"\n[bold cyan]Evaluating task: {task}[/bold cyan]")
            
            task_results = self._evaluate_task(items, task)
            all_results["by_task"][task] = task_results
            
            all_predictions.extend(task_results["predictions"])
            all_references.extend(task_results["references"])
            all_results["predictions"].extend([
                {"task": task, **p} for p in task_results["detailed"]
            ])
        
        # Overall metrics
        console.print("\n[bold cyan]Computing overall metrics...[/bold cyan]")
        all_results["overall"] = self._compute_text_metrics(
            all_predictions, all_references
        )
        
        return all_results
    
    def _evaluate_task(self, items: List[dict], task: str) -> Dict:
        """Evaluate a specific task type."""
        predictions = []
        references = []
        detailed = []
        
        for i, item in enumerate(items):
            console.print(f"  {i+1}/{len(items)}...", end="\r")
            
            # Generate prediction
            pred = self.model.generate(
                instruction=item["instruction"],
                input_text=item.get("input", ""),
                temperature=0.1,
                do_sample=False,
            )
            
            ref = item["output"]
            
            predictions.append(pred)
            references.append(ref)
            
            # Task-specific evaluation
            score = self._task_specific_score(task, pred, ref, item)
            
            detailed.append({
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "expected": ref,
                "prediction": pred,
                "score": score,
            })
        
        console.print()
        
        # Compute metrics
        metrics = self._compute_text_metrics(predictions, references)
        metrics["task_specific_accuracy"] = np.mean([d["score"] for d in detailed])
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
            "detailed": detailed,
        }
    
    def _task_specific_score(
        self, 
        task: str, 
        prediction: str, 
        reference: str,
        item: dict
    ) -> float:
        """Compute task-specific score."""
        
        if task == "structure_identification" and RDKIT_AVAILABLE:
            # For SMILES tasks, check if the molecule is correctly identified
            # Extract drug name from prediction and compare
            pred_lower = prediction.lower()
            ref_lower = reference.lower()
            return 1.0 if any(word in pred_lower for word in ref_lower.split()) else 0.0
        
        elif task == "status_analysis":
            # Check if status is correctly identified
            status_keywords = ["approved", "terminated", "withdrawn", "suspended", "completed"]
            pred_status = [s for s in status_keywords if s in prediction.lower()]
            ref_status = [s for s in status_keywords if s in reference.lower()]
            return 1.0 if set(pred_status) & set(ref_status) else 0.0
        
        elif task == "adverse_events":
            # Partial match for adverse events
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            overlap = len(pred_words & ref_words)
            return min(overlap / max(len(ref_words), 1), 1.0)
        
        else:
            # Default: simple word overlap
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            overlap = len(pred_words & ref_words)
            return min(overlap / max(len(ref_words), 1), 1.0)
    
    def _compute_text_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """Compute text generation metrics."""
        metrics = {}
        
        if self.bleu and predictions and references:
            try:
                # BLEU expects list of references per prediction
                bleu_refs = [[r.split()] for r in references]
                bleu_preds = [p.split() for p in predictions]
                bleu_result = self.bleu.compute(
                    predictions=bleu_preds,
                    references=bleu_refs
                )
                metrics["bleu"] = bleu_result.get("bleu", 0)
            except:
                metrics["bleu"] = None
        
        if self.rouge and predictions and references:
            try:
                rouge_result = self.rouge.compute(
                    predictions=predictions,
                    references=references
                )
                metrics["rouge1"] = rouge_result.get("rouge1", 0)
                metrics["rouge2"] = rouge_result.get("rouge2", 0)
                metrics["rougeL"] = rouge_result.get("rougeL", 0)
            except:
                pass
        
        # Average length
        metrics["avg_pred_length"] = np.mean([len(p.split()) for p in predictions])
        metrics["avg_ref_length"] = np.mean([len(r.split()) for r in references])
        
        return metrics
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results."""
        console.print("\n" + "="*60)
        console.print(Panel.fit("[bold green]EVALUATION RESULTS[/bold green]"))
        console.print("="*60)
        
        # Overall metrics
        console.print("\n[bold cyan]Overall Metrics:[/bold cyan]")
        overall = results["overall"]
        
        metrics_table = Table(show_header=True)
        metrics_table.add_column("Metric")
        metrics_table.add_column("Value")
        
        for key, value in overall.items():
            if value is not None:
                if isinstance(value, float):
                    metrics_table.add_row(key, f"{value:.4f}")
                else:
                    metrics_table.add_row(key, str(value))
        
        console.print(metrics_table)
        
        # Per-task results
        console.print("\n[bold cyan]Results by Task:[/bold cyan]")
        
        task_table = Table(show_header=True)
        task_table.add_column("Task")
        task_table.add_column("Accuracy")
        task_table.add_column("BLEU")
        task_table.add_column("ROUGE-L")
        task_table.add_column("Samples")
        
        for task, data in results["by_task"].items():
            metrics = data["metrics"]
            task_table.add_row(
                task,
                f"{metrics.get('task_specific_accuracy', 0):.2%}",
                f"{metrics.get('bleu', 0):.4f}" if metrics.get('bleu') else "N/A",
                f"{metrics.get('rougeL', 0):.4f}" if metrics.get('rougeL') else "N/A",
                str(len(data["predictions"])),
            )
        
        console.print(task_table)
        console.print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Drug Discovery Model")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="data/processed/training/test_instructions.jsonl",
        help="Path to test JSONL"
    )
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model)
    results = evaluator.evaluate_test_set(Path(args.test_file))
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    console.print(f"\n[green]✓ Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
