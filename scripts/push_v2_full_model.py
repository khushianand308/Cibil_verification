from unsloth import FastLanguageModel
import os

# Configuration
MERGED_MODEL_PATH = "models/cibil_qwen2.5_merged_v2"
# Pushing to the ORIGINAL repository on branch v2
REPO_NAME = "khushianand01/cibil-verification-qwen2.5-lora"
VERSION_MSG = "Full 16-bit standalone model v2.0"
BRANCH = "v2"

def main():
    if not os.path.exists(MERGED_MODEL_PATH):
        print(f"Error: Merged model not found at {MERGED_MODEL_PATH}")
        return

    print(f"Loading merged model from {MERGED_MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MERGED_MODEL_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = False, # Loading full 16-bit for upload
    )

    print(f"Pushing FULL merged model to {REPO_NAME} on branch '{BRANCH}'...")
    print("This will take 30-45 minutes as it is 15GB.")
    model.push_to_hub(REPO_NAME, commit_message = VERSION_MSG, revision = BRANCH)

    print(f"\nFull V2 Model pushed successfully to {REPO_NAME}!")

if __name__ == "__main__":
    main()
