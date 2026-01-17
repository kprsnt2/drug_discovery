"""
Enhanced Evaluation Metrics for Drug Discovery Model

Additional metrics beyond basic BLEU/ROUGE:
- Exact Match
- F1 Score (word-level)
- SMILES Validity
- Semantic Similarity
- Task-specific accuracy

Usage:
    from enhanced_metrics import EnhancedMetrics
    metrics = EnhancedMetrics()
    scores = metrics.compute_all(predictions, references)
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np

# Try to import optional libraries
try:
    from evaluate import load as load_metric
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EnhancedMetrics:
    """Comprehensive evaluation metrics for drug discovery models."""
    
    def __init__(self, use_semantic: bool = True):
        """
        Initialize metrics.
        
        Args:
            use_semantic: Whether to compute semantic similarity (slower, requires model)
        """
        self.use_semantic = use_semantic
        
        # Load evaluation metrics
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
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = None
        if use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                pass
    
    def compute_all(
        self, 
        predictions: List[str], 
        references: List[str],
        tasks: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute all metrics.
        
        Args:
            predictions: List of model predictions
            references: List of ground truth references
            tasks: Optional list of task types for task-specific metrics
        
        Returns:
            Dictionary with all computed metrics
        """
        results = {}
        
        # Basic metrics
        results.update(self.compute_exact_match(predictions, references))
        results.update(self.compute_f1_scores(predictions, references))
        results.update(self.compute_length_metrics(predictions, references))
        
        # BLEU/ROUGE
        if self.bleu:
            results.update(self.compute_bleu(predictions, references))
        if self.rouge:
            results.update(self.compute_rouge(predictions, references))
        
        # Semantic similarity
        if self.sentence_model:
            results.update(self.compute_semantic_similarity(predictions, references))
        
        # SMILES-specific metrics
        smiles_metrics = self.compute_smiles_metrics(predictions, references)
        if smiles_metrics:
            results.update(smiles_metrics)
        
        # Task-specific accuracy if tasks provided
        if tasks:
            results.update(self.compute_task_accuracy(predictions, references, tasks))
        
        return results
    
    def compute_exact_match(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """Compute exact match accuracy."""
        exact_matches = sum(
            1 for p, r in zip(predictions, references)
            if self._normalize_text(p) == self._normalize_text(r)
        )
        return {
            "exact_match": exact_matches / len(predictions) if predictions else 0,
            "exact_match_count": exact_matches,
        }
    
    def compute_f1_scores(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """Compute word-level F1 scores."""
        f1_scores = []
        precisions = []
        recalls = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._normalize_text(pred).split())
            ref_tokens = set(self._normalize_text(ref).split())
            
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0)
                precisions.append(0)
                recalls.append(0)
                continue
            
            common = pred_tokens & ref_tokens
            
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        
        return {
            "word_f1": np.mean(f1_scores),
            "word_precision": np.mean(precisions),
            "word_recall": np.mean(recalls),
        }
    
    def compute_length_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """Compute length-based metrics."""
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        length_ratios = [
            p / r if r > 0 else 0 
            for p, r in zip(pred_lengths, ref_lengths)
        ]
        
        return {
            "avg_pred_length": np.mean(pred_lengths),
            "avg_ref_length": np.mean(ref_lengths),
            "length_ratio": np.mean(length_ratios),
            "length_std": np.std(pred_lengths),
        }
    
    def compute_bleu(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """Compute BLEU scores."""
        try:
            bleu_refs = [[r.split()] for r in references]
            bleu_preds = [p.split() for p in predictions]
            result = self.bleu.compute(
                predictions=bleu_preds,
                references=bleu_refs
            )
            return {
                "bleu": result.get("bleu", 0),
                "bleu_1": result.get("precisions", [0])[0] if result.get("precisions") else 0,
                "bleu_2": result.get("precisions", [0, 0])[1] if len(result.get("precisions", [])) > 1 else 0,
            }
        except Exception as e:
            return {"bleu": None, "bleu_1": None, "bleu_2": None}
    
    def compute_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """Compute ROUGE scores."""
        try:
            result = self.rouge.compute(
                predictions=predictions,
                references=references
            )
            return {
                "rouge1": result.get("rouge1", 0),
                "rouge2": result.get("rouge2", 0),
                "rougeL": result.get("rougeL", 0),
                "rougeLsum": result.get("rougeLsum", 0),
            }
        except Exception as e:
            return {"rouge1": None, "rouge2": None, "rougeL": None}
    
    def compute_semantic_similarity(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """Compute semantic similarity using sentence embeddings."""
        if not self.sentence_model:
            return {}
        
        try:
            pred_embeddings = self.sentence_model.encode(predictions)
            ref_embeddings = self.sentence_model.encode(references)
            
            similarities = [
                cosine_similarity([p], [r])[0][0]
                for p, r in zip(pred_embeddings, ref_embeddings)
            ]
            
            return {
                "semantic_similarity": np.mean(similarities),
                "semantic_similarity_std": np.std(similarities),
                "semantic_similarity_min": np.min(similarities),
                "semantic_similarity_max": np.max(similarities),
            }
        except Exception as e:
            return {"semantic_similarity": None}
    
    def compute_smiles_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Optional[Dict]:
        """Compute SMILES-specific metrics."""
        if not RDKIT_AVAILABLE:
            return None
        
        # Extract SMILES patterns from text
        smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)=#$%\\/\.]+(?:=[A-Za-z0-9@+\-\[\]\(\)=#$%\\/\.]+)*'
        
        valid_pred_smiles = 0
        valid_ref_smiles = 0
        matching_smiles = 0
        total_smiles_found = 0
        
        for pred, ref in zip(predictions, references):
            # Find potential SMILES in predictions
            pred_candidates = re.findall(smiles_pattern, pred)
            ref_candidates = re.findall(smiles_pattern, ref)
            
            for candidate in pred_candidates:
                if len(candidate) > 5:  # Minimum reasonable SMILES length
                    mol = Chem.MolFromSmiles(candidate)
                    if mol:
                        valid_pred_smiles += 1
                        total_smiles_found += 1
            
            for candidate in ref_candidates:
                if len(candidate) > 5:
                    mol = Chem.MolFromSmiles(candidate)
                    if mol:
                        valid_ref_smiles += 1
        
        if total_smiles_found == 0:
            return None
        
        return {
            "smiles_validity_rate": valid_pred_smiles / total_smiles_found if total_smiles_found > 0 else 0,
            "valid_smiles_count": valid_pred_smiles,
            "reference_smiles_count": valid_ref_smiles,
        }
    
    def compute_task_accuracy(
        self,
        predictions: List[str],
        references: List[str],
        tasks: List[str]
    ) -> Dict:
        """Compute task-specific accuracy."""
        task_scores = {}
        task_counts = Counter(tasks)
        
        for task in set(tasks):
            indices = [i for i, t in enumerate(tasks) if t == task]
            task_preds = [predictions[i] for i in indices]
            task_refs = [references[i] for i in indices]
            
            # Compute task-specific F1
            f1_result = self.compute_f1_scores(task_preds, task_refs)
            task_scores[f"{task}_f1"] = f1_result["word_f1"]
            task_scores[f"{task}_count"] = len(indices)
        
        return task_scores
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def print_summary(self, results: Dict):
        """Print a summary of results."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        table = Table(title="Evaluation Metrics Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Priority order for display
        priority_metrics = [
            "exact_match", "word_f1", "bleu", "rouge1", "rougeL",
            "semantic_similarity", "smiles_validity_rate"
        ]
        
        for metric in priority_metrics:
            if metric in results and results[metric] is not None:
                value = results[metric]
                if isinstance(value, float):
                    table.add_row(metric, f"{value:.4f}")
                else:
                    table.add_row(metric, str(value))
        
        # Add remaining metrics
        for key, value in results.items():
            if key not in priority_metrics and value is not None:
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
        
        console.print(table)


def main():
    """Test the metrics with sample data."""
    predictions = [
        "Metformin works by decreasing glucose production in the liver.",
        "The SMILES notation for Aspirin is CC(=O)OC1=CC=CC=C1C(=O)O",
        "Common side effects include nausea and stomach pain.",
    ]
    
    references = [
        "Metformin reduces hepatic glucose production and improves insulin sensitivity.",
        "Aspirin has the SMILES structure: CC(=O)Oc1ccccc1C(=O)O",
        "Side effects may include nausea, stomach upset, and gastrointestinal issues.",
    ]
    
    tasks = ["drug_info", "structure_analysis", "adverse_events"]
    
    metrics = EnhancedMetrics(use_semantic=False)  # Set to True if you have sentence-transformers
    results = metrics.compute_all(predictions, references, tasks)
    metrics.print_summary(results)


if __name__ == "__main__":
    main()
