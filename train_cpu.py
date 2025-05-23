import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Trainer
from peft import get_peft_model, LoraConfig, TaskType, IA3Model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small model on CPU")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to dataset file (.jsonl or text)")
    parser.add_argument("--output_dir", type=str, default="~/Desktop/fine-tuned-model", help="Output directory")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA adapter into base model after training")
    
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
    
    # Format the prompts in a causal language modeling format
    def format_prompt(example):
        instruction = example["instruction"]
        response = example["output"]
        
        # Format following a typical instruction format
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return {"text": formatted_text}
    
    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)
    
    print("Loading model...")
    
    # Try TinyLlama which is specifically designed for resource-constrained environments
    print(f"Using CPU for training with model: {args.model_name}")
    
    print(f"Loading model...")
    # Load model with options to conserve memory
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Standard precision for CPU
        low_cpu_mem_usage=True,     # Conserve memory
    )
    
    print("Model loaded successfully!")
    
    # Configure LoRA
    target_modules = ["q_proj", "v_proj"]  # Simplified target for smaller models
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    # Apply LoRA to the model
    print("Applying LoRA configuration...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare the data for causal language modeling
    def preprocess_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
        
        # Create the training labels (shifted input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
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
        save_total_limit=2,
        no_cuda=True,  # Force CPU training
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,  # We're not doing masked language modeling
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training on CPU...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")

    # Optionally merge LoRA adapter into base model
    if getattr(args, 'merge_lora', False):
        print("Merging LoRA adapter into base model...")
        merged_model = IA3Model.merge_and_unload(model)
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")

if __name__ == "__main__":
    main()