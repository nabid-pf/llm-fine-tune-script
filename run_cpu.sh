#!/bin/bash

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 is required but not found. Please install Python 3.11."
    exit 1
fi

# Create and activate virtual environment with Python 3.11
if [ ! -d "train-model-env" ]; then
    echo "Creating virtual environment with Python 3.11..."
    python3.11 -m venv train-model-env
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source train-model-env/bin/activate

# Update pip3
echo "Updating pip3..."
pip3 install --upgrade pip3

# Install dependencies for CPU-only training
echo "Installing dependencies..."
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers
pip3 install datasets
pip3 install accelerate
pip3 install --upgrade peft 
pip3 install trl
pip3 install pypdf

# Verify installation
echo "Checking installed packages..."
pip3 list | grep -E 'torch|transformers|pypdf|peft|trl'

# Process PDF file if specified
if [ -n "$PDF_FILE" ]; then
    echo "Processing PDF file: $PDF_FILE"
    python3 prepare_dataset_pdf.py --input_file "$PDF_FILE" --split_strategy paragraphs
    DATASET_FILE="./data/processed_pdf_dataset.jsonl"
else
    # Prepare dataset from Ques/Ans text file
    echo "Preparing dataset from source-data.txt (Ques/Ans format)"
    python3 prepare_dataset.py --input_file source-data.txt --format ques_ans
    DATASET_FILE="./data/processed_dataset.jsonl"
    echo "No PDF file specified. Using text dataset: $DATASET_FILE"
    echo "If you want to use a PDF file, run: export PDF_FILE=path/to/your/file.pdf"
fi

# Start training with CPU-optimized script
echo "Starting CPU fine-tuning with TinyLlama model..."
python3 train_cpu.py \
    --dataset_file "$DATASET_FILE" \
    --output_dir ~/Desktop/fine-tuned-model \
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --merge_lora
    # --max_seq_length 512 \
    # --epochs 2 \
    # --batch_size 1 \
    # --gradient_accumulation_steps 4 \
    # --learning_rate 2e-4

echo "Fine-tuning complete! You can test your model with 'python3 inference.py'"
echo ""
echo "To deactivate the virtual environment when done, run: deactivate"