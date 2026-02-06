from unsloth import FastLanguageModel
import os

# Configuration
LORA_PATH = "outputs/cibil_qwen2.5_lora_v2"
# The original repository
ORIGINAL_REPO = "khushianand01/cibil-verification-qwen2.5-lora"
VERSION_MSG = "Production release v2.0 - 90% accuracy, 100% JSON validity"

def main():
    print(f"Loading v2 model from {LORA_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LORA_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    print(f"Pushing LoRA adapters to {ORIGINAL_REPO} on branch 'v2'...")
    # Pushing to the ORIGINAL repo, specifically on branch v2
    model.push_to_hub(ORIGINAL_REPO, commit_message = VERSION_MSG, revision = "v2")

    print(f"\nV2 Adapters pushed successfully to {ORIGINAL_REPO} (branch: v2)!")

if __name__ == "__main__":
    main()
