import os
import json
from datasets import Dataset

def convert_text_to_dataset(text_file, format_type=None):
    """
    Convert a text file to a dataset format suitable for fine-tuning.
    
    Args:
        text_file (str): Path to the text file
        format_type (str): Format of the text file. Options:
            - "line_by_line": Each line contains an instruction and output separated by "|||"
            - "paragraph": Paragraphs separated by blank lines, with first paragraph as instruction
            - "ques_ans": Alternating lines of Ques-XX: and Ans-XX:
    Returns:
        Dataset: Hugging Face dataset object
    """
    examples = []

    # Auto-detect format if not provided
    if format_type is None:
        if text_file.endswith('-data.txt') or os.path.basename(text_file) == 'source-data.txt':
            format_type = 'ques_ans'
        else:
            format_type = 'line_by_line'

    with open(text_file, 'r', encoding='utf-8') as f:
        if format_type == "line_by_line":
            for line in f:
                if '|||' in line:
                    parts = line.strip().split('|||')
                    if len(parts) >= 2:
                        instruction = parts[0].strip()
                        output = parts[1].strip()
                        examples.append({
                            "instruction": instruction,
                            "output": output
                        })
        elif format_type == "paragraph":
            content = f.read()
            samples = content.split("\n\n")
            for i in range(0, len(samples), 2):
                if i + 1 < len(samples):
                    instruction = samples[i].strip()
                    output = samples[i+1].strip()
                    examples.append({
                        "instruction": instruction,
                        "output": output
                    })
        elif format_type == "ques_ans":
            lines = [line.strip() for line in f if line.strip()]
            i = 0
            while i < len(lines) - 1:
                if lines[i].startswith('Ques-') and lines[i+1].startswith('Ans-'):
                    instruction = lines[i].split(':', 1)[1].strip()
                    output = lines[i+1].split(':', 1)[1].strip()
                    examples.append({
                        "instruction": instruction,
                        "output": output
                    })
                    i += 2
                else:
                    i += 1  # Skip malformed pairs

    # Save as jsonl file for later use
    os.makedirs("./data", exist_ok=True)
    with open("./data/processed_dataset.jsonl", 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    # Create dataset
    return Dataset.from_dict({
        "instruction": [ex["instruction"] for ex in examples],
        "output": [ex["output"] for ex in examples]
    })

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert text file to dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Path to text file")
    parser.add_argument("--format", type=str, default=None, 
                        choices=[None, "line_by_line", "paragraph", "ques_ans"], 
                        help="Format of the text file")
    
    args = parser.parse_args()
    
    print(f"Processing {args.input_file} with format {args.format}")
    dataset = convert_text_to_dataset(args.input_file, args.format)
    print(f"Created dataset with {len(dataset)} examples")
    print("Sample example:")
    print(dataset[0])