#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

# Check for required packages and install them if missing
required_packages = ['torch', 'transformers']

def check_and_install_packages():
    """Check for required packages and install any that are missing"""
    packages_to_install = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            packages_to_install.append(package)
            print(f"✗ {package} is not installed")
    
    if packages_to_install:
        print("\nInstalling missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
        print("Package installation complete.\n")

# Call the function to check and install packages
check_and_install_packages()

# Now we can safely import the required packages
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Add PEFT import
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned DeepSeek model")
    parser.add_argument("--model_path", type=str, default="~/Desktop/fine-tuned-model", help="Path to fine-tuned model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (if using PEFT)")
    parser.add_argument("--input", type=str, default="What is artificial intelligence?", help="Input text")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Expand home directory if used
    args.model_path = os.path.expanduser(args.model_path)
    if args.lora_path:
        args.lora_path = os.path.expanduser(args.lora_path)
    
    # Check if merged model exists
    merged_path = os.path.join(args.model_path, "merged")
    model_path = merged_path if os.path.exists(merged_path) else args.model_path
    
    print(f"Loading model from {model_path}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # If LoRA adapter is provided, load it
    if args.lora_path:
        if PeftModel is None:
            raise ImportError("peft is not installed. Please install it with 'pip install peft'.")
        print(f"Applying LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    def generate_response(instruction):
        prompt = f"""### Instruction:\n{instruction}\n\n### Response:\n"""
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print("Generating response...")
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part after "### Response:"
        response = full_response.split("### Response:")[1].strip()
        print(f"Response: {response}")
        return response
    
    # Test with example inputs
    print(f"Input: {args.input}")
    generate_response(args.input)

if __name__ == "__main__":
    main()