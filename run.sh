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

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install torch first
echo "Installing torch..."
pip install torch

# Install bitsandbytes first and make sure it's the latest version
echo "Installing latest bitsandbytes..."
pip uninstall -y bitsandbytes
pip install -U bitsandbytes --no-cache-dir

# Install other dependencies one by one
echo "Installing remaining dependencies..."
pip install "transformers>=4.30.0"
pip install datasets
pip install accelerate
pip install peft
pip install trl
pip install pypdf

# Install unsloth (optional, will skip if it fails)
echo "Trying to install Unsloth (optional)..."
pip install unsloth || echo "Unsloth installation failed, will use standard training instead."

# Check if Unsloth was installed
if python -c "import unsloth" 2>/dev/null; then
    echo "Unsloth installed successfully. Will use optimized training."
    USE_UNSLOTH=true
else
    echo "Unsloth not available. Will use standard training."
    USE_UNSLOTH=false
fi

# Verify installation
echo "Checking installed packages..."
pip list | grep -E 'unsloth|torch|transformers|pypdf|peft|trl|bitsandbytes'

# Process PDF file if specified
if [ -n "$PDF_FILE" ]; then
    DATASET_FILE="./data/processed_pdf_dataset.jsonl"
    if [ -f "$DATASET_FILE" ]; then
        echo "Found existing processed dataset at $DATASET_FILE. Skipping PDF processing."
    else
        echo "Processing PDF file: $PDF_FILE"
        python prepare_dataset_pdf.py --input_file "$PDF_FILE" --split_strategy paragraphs
    fi
else
    # Use default text dataset
    DATASET_FILE="your_data.txt"
    echo "No PDF file specified. Using text dataset: $DATASET_FILE"
    echo "If you want to use a PDF file, run: export PDF_FILE=path/to/your/file.pdf"
fi

# Start training
echo "Starting fine-tuning..."
if [ "$USE_UNSLOTH" = true ]; then
    echo "Using Unsloth for optimized training..."
    python train.py \
        --dataset_file "$DATASET_FILE" \
        --output_dir ~/Desktop/fine-tuned-model \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --max_seq_length 2048 \
        --load_in_4bit \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4
else
    echo "Using standard HuggingFace training (without 4-bit quantization)..."
    # Run without 4-bit quantization since we had issues
    python train_standard.py \
        --dataset_file "$DATASET_FILE" \
        --output_dir ~/Desktop/fine-tuned-model \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --max_seq_length 2048 \
        --epochs 3 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4
fi

echo "Fine-tuning complete! You can test your model with 'python inference.py'"
echo ""
echo "To deactivate the virtual environment when done, run: deactivate"