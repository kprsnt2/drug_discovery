"""
Drug Discovery Model Inference Script

Run inference on the fine-tuned GPT-OSS-20B model for drug discovery tasks.

Usage:
    python inference.py --model ./checkpoints/gpt-oss-20b-drug-discovery/final
    python inference.py --model ./checkpoints/gpt-oss-20b-drug-discovery/final --interactive
    python inference.py --model ./checkpoints/gpt-oss-20b-drug-discovery/final --batch input.txt
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import List, Optional

# AMD environment
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


class DrugDiscoveryModel:
    """Inference wrapper for the drug discovery model."""
    
    def __init__(
        self, 
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 512,
    ):
        """
        Initialize the model.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use ("auto", "cuda", "cpu")
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        console.print(f"[bold blue]Loading model from {model_path}...[/bold blue]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Check device
        self.device = next(self.model.parameters()).device
        console.print(f"[green]✓ Model loaded on {self.device}[/green]")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format instruction into prompt."""
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
"""
    
    def generate(
        self, 
        instruction: str, 
        input_text: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response for the given instruction.
        
        Args:
            instruction: The instruction/question
            input_text: Optional additional context
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated response text
        """
        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def batch_generate(
        self, 
        queries: List[dict],
        **kwargs
    ) -> List[dict]:
        """
        Generate responses for multiple queries.
        
        Args:
            queries: List of {"instruction": str, "input": str} dicts
            
        Returns:
            List of queries with added "response" field
        """
        results = []
        
        for i, query in enumerate(queries):
            console.print(f"Processing {i+1}/{len(queries)}...", end="\r")
            
            response = self.generate(
                instruction=query.get("instruction", ""),
                input_text=query.get("input", ""),
                **kwargs
            )
            
            results.append({
                **query,
                "response": response
            })
        
        console.print()
        return results


def interactive_mode(model: DrugDiscoveryModel):
    """Run interactive query mode."""
    console.print(Panel.fit(
        "[bold cyan]Drug Discovery AI - Interactive Mode[/bold cyan]\n\n"
        "Ask questions about drugs, structures, adverse events, and more.\n"
        "Type 'quit' or 'exit' to end the session.",
        title="🧬 Drug Discovery AI"
    ))
    
    example_queries = [
        "What is the approval status of Metformin?",
        "What are the known adverse reactions for Aspirin?",
        "Identify this drug: CC(=O)OC1=CC=CC=C1C(=O)O",
        "Why do some drugs fail clinical trials?",
    ]
    
    console.print("\n[dim]Example queries:[/dim]")
    for q in example_queries:
        console.print(f"  [dim]• {q}[/dim]")
    console.print()
    
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
            
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            if not user_input.strip():
                continue
            
            # Check if there's additional context
            context = ""
            if "|" in user_input:
                parts = user_input.split("|", 1)
                user_input = parts[0].strip()
                context = parts[1].strip()
            
            console.print("[dim]Generating...[/dim]")
            
            response = model.generate(
                instruction=user_input,
                input_text=context,
                temperature=0.7,
            )
            
            console.print(f"\n[bold blue]AI:[/bold blue] {response}\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break


def evaluate_on_test_set(model: DrugDiscoveryModel, test_file: Path, output_file: Path):
    """
    Evaluate model on test set.
    
    Args:
        model: The model to evaluate
        test_file: Path to test JSONL file
        output_file: Path to save results
    """
    console.print(f"[bold blue]Evaluating on {test_file}...[/bold blue]")
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    console.print(f"  Loaded {len(test_data)} test samples")
    
    # Generate predictions
    results = []
    correct = 0
    total = 0
    
    for i, item in enumerate(test_data):
        console.print(f"  Evaluating {i+1}/{len(test_data)}...", end="\r")
        
        prediction = model.generate(
            instruction=item["instruction"],
            input_text=item.get("input", ""),
            temperature=0.1,  # Low temperature for evaluation
            do_sample=False,
        )
        
        expected = item["output"]
        
        # Simple exact match (can be improved)
        is_correct = prediction.strip().lower() == expected.strip().lower()
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "expected": expected,
            "prediction": prediction,
            "task": item.get("task", "unknown"),
            "correct": is_correct,
        })
    
    console.print()
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Group by task
    task_results = {}
    for r in results:
        task = r["task"]
        if task not in task_results:
            task_results[task] = {"correct": 0, "total": 0}
        task_results[task]["total"] += 1
        if r["correct"]:
            task_results[task]["correct"] += 1
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "total_samples": total,
            "correct": correct,
            "task_breakdown": {
                task: {
                    "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                    "correct": data["correct"],
                    "total": data["total"],
                }
                for task, data in task_results.items()
            },
            "predictions": results,
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    console.print("\n" + "="*50)
    console.print("[bold green]EVALUATION RESULTS[/bold green]")
    console.print("="*50)
    console.print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    console.print("\nBy Task:")
    for task, data in task_results.items():
        task_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        console.print(f"  {task}: {task_acc:.2%} ({data['correct']}/{data['total']})")
    console.print(f"\nResults saved to: {output_file}")
    console.print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Drug Discovery Model Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--interactive", action="store_true", help="Interactive query mode")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--input", type=str, default="", help="Additional context for query")
    parser.add_argument("--batch", type=str, help="Path to batch input file (JSONL)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--test_file", type=str, help="Path to test JSONL for evaluation")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for batch/eval")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    # Load model
    model = DrugDiscoveryModel(
        model_path=args.model,
        max_new_tokens=args.max_tokens,
    )
    
    if args.interactive:
        interactive_mode(model)
    
    elif args.query:
        response = model.generate(
            instruction=args.query,
            input_text=args.input,
            temperature=args.temperature,
        )
        console.print(f"\n[bold blue]Response:[/bold blue]\n{response}\n")
    
    elif args.batch:
        # Batch processing
        batch_data = []
        with open(args.batch, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch_data.append(json.loads(line))
        
        results = model.batch_generate(batch_data, temperature=args.temperature)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        console.print(f"[green]✓ Results saved to {args.output}[/green]")
    
    elif args.evaluate:
        test_file = Path(args.test_file or "data/processed/training/test_instructions.jsonl")
        output_file = Path(args.output)
        evaluate_on_test_set(model, test_file, output_file)
    
    else:
        console.print("[yellow]No action specified. Use --interactive, --query, --batch, or --evaluate[/yellow]")
        parser.print_help()


if __name__ == "__main__":
    main()
