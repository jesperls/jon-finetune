import json
import os
from datasets import Dataset, DatasetDict
from typing import Dict, List


def format_prompt(example: Dict) -> str:
    prompt = f"""{example['context']}"""
    
    return prompt

def prepare_dataset(train_path: str, val_path: str, format_style: str = "mistral") -> DatasetDict:
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")

    train_formatted = [{"text": format_prompt(ex)} for ex in train_data]
    val_formatted = [{"text": format_prompt(ex)} for ex in val_data]
    
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    print(f"Processed dataset saved to {output_dir}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    
    print("\n" + "="*80)
    print("Sample training example:")
    print("="*80)
    print(train_formatted[0])
    print("="*80)
    
    return dataset_dict


def main():
    if not os.path.exists('data/train.json'):
        print("Error: data/train.json not found. Please run extract_data.py first.")
        return
    
    if not os.path.exists('data/validation.json'):
        print("Error: data/validation.json not found. Please run extract_data.py first.")
        return
    
    prepare_dataset(
        'data/train.json',
        'data/validation.json',
        format_style='mistral'
    )


if __name__ == "__main__":
    main()
