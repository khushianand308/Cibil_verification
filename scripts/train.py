import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
# Model: Qwen/Qwen2.5-7B-Instruct
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" # Optimized 4bit version
TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/val.jsonl"
OUTPUT_DIR = "outputs/cibil_qwen2.5_lora_v2"
MAX_SEQ_LENGTH = 2048

def main():
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True,
    )

    # 1. Add LoRA Adapters (Expert Fix: Target QKV+O only for stability, Dropout 0.05)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 32,
        lora_dropout = 0.05, # Expert Suggestion for Qwen2.5
        bias = "none",    # Optimized to "none" for speed
        use_gradient_checkpointing = "unsloth", # 4x longer contexts
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 2. Data Preparation
    print(f"Loading datasets: {TRAIN_FILE}, {VAL_FILE}")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": VAL_FILE}, split="train")
    eval_dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": VAL_FILE}, split="test")

    # Format the prompt using the model's native chat template
    def formatting_prompts_func(examples):
        convs = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convs]
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True,)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)

    # 3. Trainer Configuration
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Do not use packing for conversation data
        args = TrainingArguments(
            per_device_train_batch_size = 1, # Memory limit (T4)
            gradient_accumulation_steps = 8, # Effective BS = 8
            warmup_ratio = 0.05,            # Expert suggestion
            max_steps = -1,                 # Using num_train_epochs instead
            num_train_epochs = 2,           # Expert suggestion (Sweet spot)
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 25,             # Expert suggestion (Reduce noise)
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",   # Expert suggestion
            seed = 3407,
            output_dir = OUTPUT_DIR,
            eval_strategy = "no",           # Disable interim evaluation
            save_strategy = "no",           # Save final only to save disk space
            report_to = "none",
        ),
    )

    # 4. Train!
    print("Starting training...")
    trainer_stats = trainer.train()
    
    # 5. Save the final model only
    print(f"Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training Completed.")

if __name__ == "__main__":
    main()
