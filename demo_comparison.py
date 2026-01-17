"""
Drug Discovery AI - Dual Model Comparison Demo

Side-by-side comparison of Base GPT-OSS-20B vs Fine-tuned model.
Shows both responses simultaneously so users can see the difference.

Usage:
    python demo_comparison.py --finetuned ./checkpoints/gpt-oss-20b-drug-discovery/final
"""

import os
import sys
import argparse
from pathlib import Path

# Set environment for AMD GPU
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global models
base_model = None
base_tokenizer = None
finetuned_model = None
finetuned_tokenizer = None

# Example prompts showcasing the difference
COMPARISON_EXAMPLES = [
    ["Can you give me a new molecule better than paracetamol?", "Topic: Novel drug design"],
    ["What is the SMILES structure of Aspirin?", "Drug: Aspirin"],
    ["What are the side effects of Metformin?", "Drug: Metformin"],
    ["Suggest a drug for treating hypertension", "Condition: Hypertension"],
    ["What drug interactions should I avoid with Warfarin?", "Drug: Warfarin"],
    ["Design a molecule for pain relief without gastric side effects", "Topic: Drug design"],
]


def load_models(base_path: str, finetuned_path: str):
    """Load both models."""
    global base_model, base_tokenizer, finetuned_model, finetuned_tokenizer
    
    print("Loading Base Model (GPT-OSS-20B)...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, padding_side="left"
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.eval()
    print("✓ Base model loaded")
    
    # Clear cache before loading second model
    torch.cuda.empty_cache()
    
    print("Loading Fine-tuned Model...")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(
        finetuned_path, trust_remote_code=True, padding_side="left"
    )
    if finetuned_tokenizer.pad_token is None:
        finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
    
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    finetuned_model.eval()
    print("✓ Fine-tuned model loaded")
    
    return True


def generate_single(model, tokenizer, instruction: str, input_text: str, max_tokens: int = 256, temperature: float = 0.7):
    """Generate response from a single model."""
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def compare_models(instruction: str, input_text: str, max_tokens: int = 256, temperature: float = 0.7):
    """Generate responses from both models for comparison."""
    if base_model is None or finetuned_model is None:
        return "⚠️ Models not loaded", "⚠️ Models not loaded"
    
    # Generate from base model
    base_response = generate_single(base_model, base_tokenizer, instruction, input_text, max_tokens, temperature)
    
    # Generate from fine-tuned model
    finetuned_response = generate_single(finetuned_model, finetuned_tokenizer, instruction, input_text, max_tokens, temperature)
    
    return base_response, finetuned_response


def create_demo():
    """Create the comparison demo interface."""
    
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .base-output {
        border-left: 4px solid #f59e0b !important;
    }
    .finetuned-output {
        border-left: 4px solid #10b981 !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Drug Discovery AI - Model Comparison") as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2.5rem; font-weight: 700;">
                🧬 Drug Discovery AI
            </h1>
            <h2 style="color: #666; font-weight: 400;">Side-by-Side Model Comparison</h2>
            <p style="color: #888;">See the difference between Base GPT-OSS-20B and the Fine-tuned model</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                instruction = gr.Textbox(
                    label="🔬 Your Question",
                    placeholder="Ask about drugs, molecules, side effects, or request new molecule designs...",
                    lines=2,
                )
                input_text = gr.Textbox(
                    label="📋 Context (optional)",
                    placeholder="Drug name, condition, or additional context",
                    lines=1,
                )
                
                with gr.Row():
                    max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max Tokens")
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                
                compare_btn = gr.Button("⚡ Compare Both Models", variant="primary", size="lg")
                
                gr.Markdown("### 💡 Try These Examples")
                gr.Examples(
                    examples=COMPARISON_EXAMPLES,
                    inputs=[instruction, input_text],
                    label=""
                )
        
        gr.HTML("<hr style='margin: 2rem 0;'>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🟡 Base GPT-OSS-20B")
                base_output = gr.Textbox(
                    label="Response",
                    lines=12,
                    elem_classes=["base-output"]
                )
                gr.Markdown("*Original model without fine-tuning*")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🟢 Fine-tuned Model")
                finetuned_output = gr.Textbox(
                    label="Response", 
                    lines=12,
                    elem_classes=["finetuned-output"]
                )
                gr.Markdown("*Trained on 4,730 drug discovery samples*")
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; 
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    border-radius: 10px;">
            <p style="margin: 0; color: #666;">
                ⚠️ <strong>Disclaimer:</strong> For research and educational purposes only. 
                Not for medical advice. | 
                <a href="https://github.com/kprsnt2/drug_discovery">GitHub</a> | 
                <a href="https://kprsnt.in">kprsnt.in</a>
            </p>
        </div>
        """)
        
        compare_btn.click(
            fn=compare_models,
            inputs=[instruction, input_text, max_tokens, temperature],
            outputs=[base_output, finetuned_output]
        )
        
        instruction.submit(
            fn=compare_models,
            inputs=[instruction, input_text, max_tokens, temperature],
            outputs=[base_output, finetuned_output]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Drug Discovery AI - Dual Model Comparison Demo")
    parser.add_argument(
        "--base",
        type=str,
        default="openai/gpt-oss-20b",
        help="Path or HuggingFace ID for base model"
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        default="./checkpoints/gpt-oss-20b-drug-discovery/final",
        help="Path to fine-tuned model"
    )
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Drug Discovery AI - Dual Model Comparison")
    print("=" * 60)
    
    if not load_models(args.base, args.finetuned):
        print("Failed to load models!")
        sys.exit(1)
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
