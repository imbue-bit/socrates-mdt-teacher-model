#!/usr/bin/env python3
"""
SFT to Pre-training Data Converter

Converts Supervised Fine-Tuning datasets to plain text format for BERT pre-training.
Specifically designed for nvidia/OpenScience dataset.

Usage:
    python sft2pt.py --dataset nvidia/OpenScience --output_file sft_text.txt --split train
"""

import argparse
from datasets import load_dataset
import os

def convert_sft_to_text(dataset_name, config_name, split, output_file, max_samples=None):
    """
    Convert SFT dataset to plain text for pre-training.

    Args:
        dataset_name: HuggingFace dataset name
        config_name: Dataset config name
        split: Dataset split (train, test, etc.)
        output_file: Output text file path
        max_samples: Maximum number of samples to process (None for all)
    """
    print(f"Loading dataset: {dataset_name}, config: {config_name}, split: {split}")

    # Load dataset
    dataset = load_dataset(dataset_name, config_name, split=split)

    print(f"Dataset loaded. Total samples: {len(dataset)}")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using first {len(dataset)} samples")

    # Convert to text
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(dataset):
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(dataset)}")

            # Extract input and output
            input_text = sample.get('input', '').strip()
            output_text = sample.get('output', '').strip()

            # Combine input and output as continuous text
            if input_text and output_text:
                # Add separator between input and output
                combined_text = f"{input_text}\n\n{output_text}"
            elif input_text:
                combined_text = input_text
            elif output_text:
                combined_text = output_text
            else:
                continue  # Skip empty samples

            # Write to file with double newline as document separator
            f.write(combined_text + '\n\n')

    print(f"Conversion complete. Output saved to: {output_file}")

    # Calculate some statistics
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        total_chars = len(content)
        total_lines = content.count('\n')
        approx_tokens = total_chars // 4  # Rough estimate: 4 chars per token

    print("Statistics:")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Approximate tokens: {approx_tokens:,}")
    print(f"  Number of text blocks: {total_lines // 2}")  # Each sample has 2 newlines

def main():
    parser = argparse.ArgumentParser(description="Convert SFT dataset to pre-training text")
    parser.add_argument("--dataset", type=str, default="nvidia/OpenScience",
                       help="HuggingFace dataset name")
    parser.add_argument("--config", type=str, default="OS-Q3-235B-4",
                       help="Dataset config name")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--output_file", type=str, default="sft_text.txt",
                       help="Output text file path")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    convert_sft_to_text(args.dataset, args.config, args.split, args.output_file, args.max_samples)

if __name__ == "__main__":
    main()