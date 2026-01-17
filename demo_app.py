"""
Drug Discovery AI - Gradio Demo Interface

A beautiful web UI for interacting with the fine-tuned drug discovery model.

Usage:
    python demo_app.py --model ./checkpoints/gpt-oss-20b-drug-discovery/final
    python demo_app.py --share  # Create public link
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

# Global model and tokenizer
model = None
tokenizer = None

# Example prompts for each task
EXAMPLE_PROMPTS = {
    "Drug Information": [
        ("What is the mechanism of action of Metformin?", "Drug: Metformin"),
        ("Describe how Atorvastatin works.", "Drug: Atorvastatin"),
        ("What therapeutic class does Lisinopril belong to?", "Drug: Lisinopril"),
    ],
    "Adverse Events": [
        ("What are the common side effects of Ibuprofen?", "Drug: Ibuprofen"),
        ("What serious reactions are associated with Fluoroquinolones?", "Drug class: Fluoroquinolones"),
        ("List potential adverse effects of Methotrexate.", "Drug: Methotrexate"),
    ],
    "SMILES & Structure": [
        ("What is the SMILES notation for Aspirin?", "Drug: Aspirin"),
        ("Describe the molecular structure of Caffeine.", "Drug: Caffeine"),
        ("What functional groups are in Acetaminophen?", "Drug: Acetaminophen"),
    ],
    "Drug Interactions": [
        ("Can Warfarin be taken with Aspirin?", "Drug combination: Warfarin + Aspirin"),
        ("What happens when MAO inhibitors are combined with SSRIs?", "Drug combination: MAOIs + SSRIs"),
        ("Are there interactions between Metformin and alcohol?", "Drug: Metformin + Alcohol"),
    ],
    "Clinical & FDA": [
        ("What is the FDA approval status of Pembrolizumab?", "Drug: Pembrolizumab (Keytruda)"),
        ("What clinical trial phases are required for approval?", "Topic: FDA approval process"),
        ("Is Remdesivir approved for COVID-19?", "Drug: Remdesivir"),
    ],
}


def load_model(model_path: str):
    """Load the model and tokenizer."""
    global model, tokenizer
    
    print(f"Loading model from: {model_path}")
    
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
    
    print("Model loaded successfully!")
    return True


def generate_response(
    instruction: str,
    input_text: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response from the model."""
    global model, tokenizer
    
    if model is None:
        return "⚠️ Model not loaded. Please load a model first."
    
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
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def create_demo():
    """Create the Gradio demo interface."""
    
    # Custom CSS for beautiful styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .example-btn {
        margin: 0.25rem;
    }
    .output-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Drug Discovery AI", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🧬 Drug Discovery AI</h1>
            <p style="color: #666; font-size: 1.1rem;">
                Fine-tuned GPT-OSS-20B for pharmaceutical and drug discovery tasks
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📝 Query")
                
                instruction = gr.Textbox(
                    label="Instruction",
                    placeholder="What would you like to know about a drug?",
                    lines=2,
                )
                
                input_text = gr.Textbox(
                    label="Input/Context",
                    placeholder="Drug: [Drug Name] or additional context",
                    lines=2,
                )
                
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=64, maximum=512, value=256, step=32,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature"
                    )
                
                submit_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")
                
                # Example buttons
                gr.Markdown("### 💡 Example Queries")
                
                with gr.Accordion("Drug Information", open=False):
                    for inst, inp in EXAMPLE_PROMPTS["Drug Information"]:
                        gr.Button(inst[:50] + "...", size="sm").click(
                            lambda i=inst, n=inp: (i, n),
                            outputs=[instruction, input_text]
                        )
                
                with gr.Accordion("Adverse Events", open=False):
                    for inst, inp in EXAMPLE_PROMPTS["Adverse Events"]:
                        gr.Button(inst[:50] + "...", size="sm").click(
                            lambda i=inst, n=inp: (i, n),
                            outputs=[instruction, input_text]
                        )
                
                with gr.Accordion("SMILES & Structure", open=False):
                    for inst, inp in EXAMPLE_PROMPTS["SMILES & Structure"]:
                        gr.Button(inst[:50] + "...", size="sm").click(
                            lambda i=inst, n=inp: (i, n),
                            outputs=[instruction, input_text]
                        )
                
                with gr.Accordion("Drug Interactions", open=False):
                    for inst, inp in EXAMPLE_PROMPTS["Drug Interactions"]:
                        gr.Button(inst[:50] + "...", size="sm").click(
                            lambda i=inst, n=inp: (i, n),
                            outputs=[instruction, input_text]
                        )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 🤖 AI Response")
                
                output = gr.Textbox(
                    label="Generated Response",
                    lines=15,
                )
                
                gr.Markdown("""
                ---
                ### ⚠️ Disclaimer
                
                This AI is for **research and educational purposes only**. 
                It should not be used for medical decision-making. 
                Always consult healthcare professionals and official drug information sources.
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <p style="color: #666; margin: 0;">
                Built with ❤️ using GPT-OSS-20B fine-tuned on AMD MI300X | 
                <a href="https://github.com/kprsnt2/drug_discovery" target="_blank">GitHub</a> | 
                <a href="https://kprsnt.in" target="_blank">kprsnt.in</a>
            </p>
        </div>
        """)
        
        # Event handlers
        submit_btn.click(
            fn=generate_response,
            inputs=[instruction, input_text, max_tokens, temperature],
            outputs=[output]
        )
        
        # Also trigger on Enter
        instruction.submit(
            fn=generate_response,
            inputs=[instruction, input_text, max_tokens, temperature],
            outputs=[output]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Drug Discovery AI Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="./checkpoints/gpt-oss-20b-drug-discovery/final",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    
    # Load model
    if not load_model(args.model):
        print("Failed to load model!")
        sys.exit(1)
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
