import os
import torch
import argparse
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

def format_prompt(example):
    """Format prompts for training"""
    instruction = example["instruction"]
    response = example["output"]
    
    prompt = f"""### Instruction:
{instruction}

### Response:
{response}"""
    
    return {"text": prompt}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek model")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to dataset file (.jsonl or text)")
    parser.add_argument("--output_dir", type=str, default="~/Desktop/fine-tuned-model", help="Output directory")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load in 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    args = parser.parse_args()
    
    # Expand home directory if used
    args.output_dir = os.path.expanduser(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading dataset...")
    if args.dataset_file.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=args.dataset_file)['train']
    else:
        # Process text file with prepare_dataset.py
        from prepare_dataset import convert_text_to_dataset
        dataset = convert_text_to_dataset(args.dataset_file)
    
    # Split dataset
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits['train']
    eval_dataset = splits['test']
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Format the prompts
    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Prepare the model for PEFT/LoRA training
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True if not args.load_in_4bit else False,
        bf16=False,
        save_total_limit=2,
        report_to=None,
    )
    
    # Configure the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        packing=False,
        dataset_text_field="text",
        # Setting this to 1 to avoid issues on Windows
        dataset_num_proc=1,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save merged model for easier inference
    print("Saving merged model...")
    model.merge_and_unload().save_pretrained(f"{args.output_dir}/merged")
    tokenizer.save_pretrained(f"{args.output_dir}/merged")
    
    print("Training complete!")

if __name__ == "__main__":
    main()