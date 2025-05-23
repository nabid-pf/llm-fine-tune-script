# DeepSeek Model Fine-tuning

This repository contains scripts to fine-tune DeepSeek models with your own data using Python 3.11 and Unsloth.

## Requirements

- Python 3.11 (Unsloth doesn't support Python 3.13)
- CUDA-capable GPU (recommended)
- Your custom dataset in text or JSONL format

## Setup

1. Make sure Python 3.11 is installed on your system. You can check with:
   ```bash
   python3.11 --version
   ```

2. If Python 3.11 is not installed, you'll need to install it first:
   - On macOS with Homebrew: `brew install python@3.11`
   - On Ubuntu/Debian: `sudo apt install python3.11 python3.11-venv`
   - On Windows: Download from [python.org](https://www.python.org/downloads/)

## Starting the Virtual Environment

You can start the virtual environment in two ways:

### Option 1: Using run.sh (recommended)
Prepare `source-data.txt` file in given format (check `sample-source-data.txt`)
Our `run.sh` script will automatically create and activate the virtual environment:

```bash
chmod +x run.sh  # Make the script executable
./run.sh         # Run the script
```

### Option 2: Manual Activation

You can also manually create and activate the virtual environment:

```bash
# Create virtual environment
python3.11 -m venv train-model-env-py311

# Activate the environment:
# On macOS/Linux:
source train-model-env-py311/bin/activate

# On Windows:
# train-model-env-py311\Scripts\activate
```

After the environment is activated, install dependencies:
```bash
pip install torch
pip install unsloth transformers datasets accelerate peft trl
```

## Dataset Preparation

Your dataset should be in one of the following formats:

1. **Line-by-line format**: Each line contains an instruction and output separated by `|||`
   ```
   What is machine learning? ||| Machine learning is a field of artificial intelligence...
   How do neural networks work? ||| Neural networks are computing systems inspired by...
   ```

2. **Paragraph format**: Alternate paragraphs for instruction and response, separated by blank lines
   ```
   What is machine learning?

   Machine learning is a field of artificial intelligence...

   How do neural networks work?

   Neural networks are computing systems inspired by...
   ```

3. **JSONL format**: Each line is a JSON object with "instruction" and "output" fields
   ```json
   {"instruction": "What is machine learning?", "output": "Machine learning is a field of artificial intelligence..."}
   {"instruction": "How do neural networks work?", "output": "Neural networks are computing systems inspired by..."}
   ```

## Training

Before running training, edit the `run.sh` file to set your dataset file path.

```bash
# Example training command
python train.py \
    --dataset_file your_data.txt \
    --output_dir ~/Desktop/fine-tuned-model \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --load_in_4bit \
    --epochs 3
```

You can adjust parameters like:
- `--model_name`: Select different DeepSeek models (8B, 70B, etc.)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for training

## Inference

After training, test your model with:

```bash
python inference.py \
    --model_path ~/Desktop/fine-tuned-model \
    --input "Your test question here" \
    --temperature 0.6
```

## Available DeepSeek Models

You can use various DeepSeek models, including:

- deepseek-ai/DeepSeek-R1-Distill-Llama-8B (default)
- deepseek-ai/DeepSeek-R1-Distill-Llama-70B (larger, requires more VRAM)
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (smallest)
- deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (better performance)

## Hardware Requirements

- For 4-bit quantization (--load_in_4bit): 8GB+ VRAM recommended
- CPU with AVX2 support
- At least 16GB of system RAM

## Troubleshooting

If you encounter dependency issues, try:

1. Installing PyTorch separately first:
   ```bash
   pip install torch
   ```

2. Then install the rest of the dependencies:
   ```bash
   pip install unsloth transformers datasets accelerate peft trl
   ```

## Deactivating the Virtual Environment

When done, deactivate the virtual environment with:
```bash
deactivate
```