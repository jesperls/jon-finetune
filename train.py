import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()


def setup_model(model_name: str, max_seq_length: int):
    print(f"Loading model from HuggingFace: {model_name}")
    
    hf_token = os.getenv('HF_TOKEN', None)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    print("Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable: {100 * trainable_params / all_params:.2f}%")
    
    return model, tokenizer


def train(
    model_name: str = None,
    max_seq_length: int = None,
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    learning_rate: float = None,
    num_epochs: int = None,
    warmup_steps: int = None,
    output_dir: str = None,
):
    model_name = model_name or os.getenv('MODEL_NAME', 'mistralai/Mistral-7B-Instruct-v0.3')
    max_seq_length = max_seq_length or int(os.getenv('MAX_SEQ_LENGTH', 256))
    batch_size = batch_size or int(os.getenv('BATCH_SIZE', 2))
    gradient_accumulation_steps = gradient_accumulation_steps or int(os.getenv('GRADIENT_ACCUMULATION_STEPS', 4))
    learning_rate = learning_rate or float(os.getenv('LEARNING_RATE', 2e-4))
    num_epochs = num_epochs or int(os.getenv('NUM_EPOCHS', 3))
    warmup_steps = warmup_steps or int(os.getenv('WARMUP_STEPS', 5))
    output_dir = output_dir or os.getenv('OUTPUT_DIR', './outputs')
    
    print("Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Output directory: {output_dir}")
    print()
    
    model, tokenizer = setup_model(model_name, max_seq_length)
    
    dataset_path = 'data/processed'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. "
            "Please run prepare_dataset.py first."
        )
    
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print(f"Dataset loaded:")
    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    print()
    
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        report_to="none",
        run_name="twitch-chat-finetune",
    )
    
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result
    
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing training data",
    )
    
    tokenized_eval = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['validation'].column_names,
        desc="Tokenizing validation data",
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("GPU Memory before training:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print()
    
    print("Starting training...")
    trainer.train()
    
    print("\nSaving final model...")
    final_output_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"Training complete! LoRA adapters saved to {final_output_dir}")
    print(f"\nTo use the model, load it with:")
    print(f"  from peft import AutoPeftModelForCausalLM")
    print(f"  model = AutoPeftModelForCausalLM.from_pretrained('{final_output_dir}')")
    
    return model, tokenizer


def main():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    train()


if __name__ == "__main__":
    main()
