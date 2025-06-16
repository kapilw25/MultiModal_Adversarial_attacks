#!/usr/bin/env python
"""
Multi-modal Self-instruct Dataset Downloader

This script downloads the Multi-modal Self-instruct dataset from Hugging Face
and saves it to disk for further use in training and evaluation.
"""

import os
import argparse
from datasets import load_dataset

def download_dataset(output_dir="./data"):
    """
    Download the Multi-modal Self-instruct dataset and save it to disk.
    
    Args:
        output_dir (str): Directory where the dataset will be saved
    """
    print(f"Downloading Multi-modal Self-instruct dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset("zwq2018/Multi-modal-Self-instruct")
    
    # Save dataset to disk
    dataset.save_to_disk(output_dir)
    
    print(f"Dataset successfully downloaded and saved to {output_dir}")
    print(f"Dataset structure:")
    print(f"- Training set: {len(dataset['train'])} examples")
    print(f"- Testing set: {len(dataset['test'])} examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Multi-modal Self-instruct dataset")
    parser.add_argument("--output_dir", type=str, default="./data", 
                        help="Directory where the dataset will be saved")
    args = parser.parse_args()
    
    download_dataset(args.output_dir)
