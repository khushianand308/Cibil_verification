from unsloth import FastLanguageModel
import os

# Configuration
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
LORA_PATH = "outputs/cibil_qwen2.5_lora_v2"
REPO_NAME = "khushianand01/cibil-verification-v2"
VERSION_MSG = "Production release v2.0 - 90% accuracy, 100% JSON validity"

def main():
    print(f"Loading v2 model from {LORA_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LORA_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    print(f"Pushing LoRA adapters to {REPO_NAME}-lora...")
    model.push_to_hub(f"{REPO_NAME}-lora", commit_message = VERSION_MSG)

    print("\nV2 Adapters pushed successfully!")

if __name__ == "__main__":
    main()
