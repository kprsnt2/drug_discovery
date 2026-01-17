"""
Model Comparison Script: Base vs Fine-tuned

Compares the base GPT-OSS-20B model with the fine-tuned drug discovery model.
Generates side-by-side comparison tables and improvement metrics.

Usage:
    python compare_models.py --base openai/gpt-oss-20b --finetuned ./checkpoints/gpt-oss-20b-drug-discovery/final
    python compare_models.py --finetuned ./checkpoints/gpt-oss-20b-drug-discovery/final  # Uses default base
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Set environment for AMD GPU
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Test prompts for comparison
COMPARISON_PROMPTS = [
    {
        "task": "drug_info",
        "instruction": "What is the mechanism of action of Metformin?",
        "input": "Drug: Metformin",
        "expected_keywords": ["diabetes", "glucose", "insulin", "liver", "blood sugar"]
    },
    {
        "task": "adverse_events",
        "instruction": "What are the common side effects of Ibuprofen?",
        "input": "Drug: Ibuprofen",
        "expected_keywords": ["stomach", "gastrointestinal", "bleeding", "nausea", "ulcer"]
    },
    {
        "task": "structure_analysis",
        "instruction": "What is the SMILES notation for Aspirin?",
        "input": "Drug: Aspirin (Acetylsalicylic acid)",
        "expected_keywords": ["CC(=O)", "c1ccccc1", "SMILES", "O=C"]
    },
    {
        "task": "drug_interaction",
        "instruction": "Can Warfarin be taken with Aspirin?",
        "input": "Drug combination: Warfarin + Aspirin",
        "expected_keywords": ["bleeding", "risk", "interaction", "caution", "anticoagulant"]
    },
    {
        "task": "indication",
        "instruction": "What conditions is Lisinopril used to treat?",
        "input": "Drug: Lisinopril",
        "expected_keywords": ["hypertension", "blood pressure", "heart", "ACE", "cardiovascular"]
    },
    {
        "task": "pharmacology",
        "instruction": "Explain the pharmacokinetics of Omeprazole.",
        "input": "Drug: Omeprazole",
        "expected_keywords": ["proton pump", "acid", "absorption", "metabolism", "half-life"]
    },
    {
        "task": "dosage",
        "instruction": "What is the typical dosage for Amoxicillin in adults?",
        "input": "Drug: Amoxicillin",
        "expected_keywords": ["mg", "daily", "oral", "infection", "antibiotic"]
    },
    {
        "task": "contraindication",
        "instruction": "Who should not take Penicillin?",
        "input": "Drug: Penicillin",
        "expected_keywords": ["allergy", "hypersensitivity", "allergic", "reaction"]
    },
]


@dataclass
class ModelResponse:
    """Container for model response and metrics."""
    text: str
    latency: float
    keyword_matches: int
    total_keywords: int
    relevance_score: float


class ModelComparator:
    """Compare base and fine-tuned models."""
    
    def __init__(self, base_model_path: str, finetuned_model_path: str):
        """Initialize with both model paths."""
        self.base_path = base_model_path
        self.finetuned_path = finetuned_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
    
    def load_model(self, model_path: str, name: str) -> Tuple:
        """Load a model and tokenizer."""
        console.print(f"[bold blue]Loading {name}...[/bold blue]")
        console.print(f"  Path: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        model.eval()
        
        console.print(f"  [green]✓ {name} loaded successfully[/green]")
        return model, tokenizer
    
    def load_models(self):
        """Load both models."""
        self.base_model, self.base_tokenizer = self.load_model(
            self.base_path, "Base Model (GPT-OSS-20B)"
        )
        
        # Clear some GPU memory before loading second model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.finetuned_model, self.finetuned_tokenizer = self.load_model(
            self.finetuned_path, "Fine-tuned Model"
        )
    
    def generate_response(
        self, 
        model, 
        tokenizer, 
        instruction: str, 
        input_text: str,
        max_new_tokens: int = 256
    ) -> Tuple[str, float]:
        """Generate response from a model."""
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        latency = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response, latency
    
    def evaluate_response(
        self, 
        response: str, 
        expected_keywords: List[str]
    ) -> Tuple[int, float]:
        """Evaluate response quality based on keyword matching."""
        response_lower = response.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        relevance = matches / len(expected_keywords) if expected_keywords else 0.0
        return matches, relevance
    
    def run_comparison(self) -> Dict:
        """Run full comparison on all test prompts."""
        results = {
            "base": [],
            "finetuned": [],
            "prompts": COMPARISON_PROMPTS
        }
        
        console.print("\n[bold yellow]Running Comparison Tests[/bold yellow]")
        console.print("=" * 60)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Comparing models...", total=len(COMPARISON_PROMPTS) * 2)
            
            for prompt_data in COMPARISON_PROMPTS:
                # Base model
                progress.update(task, description=f"Base: {prompt_data['task']}")
                base_response, base_latency = self.generate_response(
                    self.base_model,
                    self.base_tokenizer,
                    prompt_data["instruction"],
                    prompt_data["input"]
                )
                base_matches, base_relevance = self.evaluate_response(
                    base_response, prompt_data["expected_keywords"]
                )
                results["base"].append(ModelResponse(
                    text=base_response,
                    latency=base_latency,
                    keyword_matches=base_matches,
                    total_keywords=len(prompt_data["expected_keywords"]),
                    relevance_score=base_relevance
                ))
                progress.advance(task)
                
                # Fine-tuned model
                progress.update(task, description=f"Fine-tuned: {prompt_data['task']}")
                ft_response, ft_latency = self.generate_response(
                    self.finetuned_model,
                    self.finetuned_tokenizer,
                    prompt_data["instruction"],
                    prompt_data["input"]
                )
                ft_matches, ft_relevance = self.evaluate_response(
                    ft_response, prompt_data["expected_keywords"]
                )
                results["finetuned"].append(ModelResponse(
                    text=ft_response,
                    latency=ft_latency,
                    keyword_matches=ft_matches,
                    total_keywords=len(prompt_data["expected_keywords"]),
                    relevance_score=ft_relevance
                ))
                progress.advance(task)
        
        return results
    
    def generate_report(self, results: Dict) -> None:
        """Generate and print comparison report."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]Model Comparison Results[/bold green]",
            border_style="green"
        ))
        
        # Summary table
        summary_table = Table(title="Performance Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Base GPT-OSS-20B", style="yellow")
        summary_table.add_column("Fine-tuned", style="green")
        summary_table.add_column("Improvement", style="magenta")
        
        # Calculate averages
        base_avg_relevance = sum(r.relevance_score for r in results["base"]) / len(results["base"])
        ft_avg_relevance = sum(r.relevance_score for r in results["finetuned"]) / len(results["finetuned"])
        relevance_improvement = ((ft_avg_relevance - base_avg_relevance) / max(base_avg_relevance, 0.01)) * 100
        
        base_avg_latency = sum(r.latency for r in results["base"]) / len(results["base"])
        ft_avg_latency = sum(r.latency for r in results["finetuned"]) / len(results["finetuned"])
        
        base_total_matches = sum(r.keyword_matches for r in results["base"])
        ft_total_matches = sum(r.keyword_matches for r in results["finetuned"])
        total_keywords = sum(r.total_keywords for r in results["base"])
        
        summary_table.add_row(
            "Relevance Score",
            f"{base_avg_relevance:.1%}",
            f"{ft_avg_relevance:.1%}",
            f"+{relevance_improvement:.1f}%" if relevance_improvement > 0 else f"{relevance_improvement:.1f}%"
        )
        summary_table.add_row(
            "Keyword Accuracy",
            f"{base_total_matches}/{total_keywords} ({base_total_matches/total_keywords:.1%})",
            f"{ft_total_matches}/{total_keywords} ({ft_total_matches/total_keywords:.1%})",
            f"+{((ft_total_matches - base_total_matches) / max(base_total_matches, 1)) * 100:.1f}%"
        )
        summary_table.add_row(
            "Avg Response Time",
            f"{base_avg_latency:.2f}s",
            f"{ft_avg_latency:.2f}s",
            f"{((ft_avg_latency - base_avg_latency) / base_avg_latency) * 100:+.1f}%"
        )
        
        console.print(summary_table)
        
        # Per-task breakdown
        console.print("\n")
        task_table = Table(title="Per-Task Breakdown", show_header=True)
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Base Score", style="yellow")
        task_table.add_column("Fine-tuned Score", style="green")
        task_table.add_column("Winner", style="bold")
        
        for i, prompt in enumerate(results["prompts"]):
            base_score = results["base"][i].relevance_score
            ft_score = results["finetuned"][i].relevance_score
            winner = "🏆 Fine-tuned" if ft_score > base_score else ("🏆 Base" if base_score > ft_score else "Tie")
            
            task_table.add_row(
                prompt["task"],
                f"{base_score:.0%}",
                f"{ft_score:.0%}",
                winner
            )
        
        console.print(task_table)
        
        # Sample responses comparison
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]Sample Response Comparison[/bold cyan]",
            border_style="cyan"
        ))
        
        # Show first 3 comparisons
        for i in range(min(3, len(results["prompts"]))):
            prompt = results["prompts"][i]
            base_resp = results["base"][i]
            ft_resp = results["finetuned"][i]
            
            console.print(f"\n[bold]Prompt: {prompt['instruction']}[/bold]")
            console.print(f"[dim]Input: {prompt['input']}[/dim]\n")
            
            console.print("[yellow]Base Model Response:[/yellow]")
            console.print(f"  {base_resp.text[:300]}..." if len(base_resp.text) > 300 else f"  {base_resp.text}")
            console.print(f"  [dim]Score: {base_resp.relevance_score:.0%}, Time: {base_resp.latency:.2f}s[/dim]")
            
            console.print("\n[green]Fine-tuned Model Response:[/green]")
            console.print(f"  {ft_resp.text[:300]}..." if len(ft_resp.text) > 300 else f"  {ft_resp.text}")
            console.print(f"  [dim]Score: {ft_resp.relevance_score:.0%}, Time: {ft_resp.latency:.2f}s[/dim]")
            
            console.print("-" * 60)
        
        # Final verdict
        console.print("\n")
        if ft_avg_relevance > base_avg_relevance:
            verdict = f"🎉 Fine-tuned model shows [bold green]{relevance_improvement:.1f}% improvement[/bold green] in drug discovery tasks!"
        elif ft_avg_relevance == base_avg_relevance:
            verdict = "📊 Both models perform similarly on drug discovery tasks."
        else:
            verdict = "⚠️ Base model performed better. Consider reviewing fine-tuning approach."
        
        console.print(Panel.fit(verdict, title="Verdict", border_style="green"))
        
        # Save results to JSON
        output_file = Path("comparison_results.json")
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "base_relevance": base_avg_relevance,
                    "finetuned_relevance": ft_avg_relevance,
                    "improvement_percent": relevance_improvement,
                    "base_latency": base_avg_latency,
                    "finetuned_latency": ft_avg_latency,
                },
                "per_task": [
                    {
                        "task": p["task"],
                        "base_score": results["base"][i].relevance_score,
                        "finetuned_score": results["finetuned"][i].relevance_score,
                    }
                    for i, p in enumerate(results["prompts"])
                ]
            }, f, indent=2)
        
        console.print(f"\n[dim]Results saved to {output_file}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned models")
    parser.add_argument(
        "--base",
        type=str,
        default="openai/gpt-oss-20b",
        help="Path or HuggingFace ID for base model"
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]Drug Discovery Model Comparison[/bold blue]\n"
        "Base model vs Fine-tuned model benchmark",
        border_style="blue"
    ))
    
    # Check paths
    if not Path(args.finetuned).exists() and not args.finetuned.startswith(("openai/", "huggingface/")):
        console.print(f"[red]Error: Fine-tuned model path not found: {args.finetuned}[/red]")
        sys.exit(1)
    
    # Initialize comparator
    comparator = ModelComparator(args.base, args.finetuned)
    
    try:
        # Load models
        comparator.load_models()
        
        # Run comparison
        results = comparator.run_comparison()
        
        # Generate report
        comparator.generate_report(results)
        
    except Exception as e:
        console.print(f"[red]Error during comparison: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
