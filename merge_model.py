import os
from unsloth import FastLanguageModel
import torch

# --- Configuration ---
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTERS_REPO = "khushianand01/cibil-verification-qwen2.5-lora"
ADAPTERS_REVISION = "v2"
SAVE_PATH = "/home/ubuntu/Cibil_verification/models/cibil-qwen2.5-7b-v2-merged"

def merge_and_save():
    print(f"Loading base model and adapters: {ADAPTERS_REPO} (Revision: {ADAPTERS_REVISION})...")
    
    # 4-bit is necessary for T4 memory limits
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = ADAPTERS_REPO, 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    print("Merging adapters into 4-bit standalone model...")
    model.save_pretrained_merged(SAVE_PATH, tokenizer, save_method = "merged_4bit_forced")
    print(f"Success! Optimized 4-bit model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        import shutil
        shutil.rmtree(SAVE_PATH)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    merge_and_save()
