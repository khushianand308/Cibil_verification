import os
from unsloth import FastLanguageModel

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
LORA_PATH = "outputs/cibil_qwen25_lora"
MAX_SEQ_LENGTH = 2048

def main():
    print(f"Loading base model and adapters: {LORA_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LORA_PATH, # Loads the saved lora adapters
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    # 1. Merge and Save to 16bit (Float16)
    # This is the best format for vLLM or standard Hugging Face inference.
    # It combines the "base model" + "your training" into one folder.
    merged_path = "models/cibil_qwen25_merged_16bit"
    print(f"\n--- Exporting Merged 16-bit model to {merged_path} ---")
    print("This takes a few minutes and ~15GB disk space...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method = "merged_16bit")

    # 2. Export to GGUF (Optional)
    # This is the best format for Ollama, llama.cpp, or running on local laptops/CPUs.
    # We use q4_k_m (a high quality 4-bit quantization).
    gguf_path = "models/cibil_qwen25_q4_k_m.gguf"
    print(f"\n--- Exporting GGUF (4-bit) to {gguf_path} ---")
    # Note: save_pretrained_gguf might require llama.cpp installed or auto-installs it.
    try:
        model.save_pretrained_gguf(
            "models/cibil_qwen25_gguf", 
            tokenizer, 
            quantization_method = "q4_k_m"
        )
    except Exception as e:
        print(f"⚠️ GGUF export failed or skipped: {e}")
        print("Install llama.cpp or ensure enough RAM for conversion.")

    print("\n✅ Export Process Completed.")
    print(f"Merged model: {merged_path}")
    print(f"GGUF models (if successful) are in: models/cibil_qwen25_gguf/")

    # 3. PUSH TO HUGGING FACE
    # We push both the lightweight adapters and the full production-ready merged model.
    # Versioning: Every push creates a new commit. We'll label this as v1.0.
    repo_name = "khushianand01/cibil-verification-qwen2.5"
    version_msg = "Initial production release v1.0"
    
    print(f"\n--- Pushing to Hugging Face: {repo_name} ---")
    
    # Push Adapters (Option 2) - This is fast (~40MB)
    print("Pushing LoRA adapters...")
    model.push_to_hub(f"{repo_name}-lora", commit_message = version_msg) 
    
    # Push Merged 16bit (Option 1) - SKIPPED as per user request
    # print("Pushing merged 16-bit model (this may take a while)...")
    # model.push_to_hub_merged(repo_name, tokenizer, save_method = "merged_16bit", commit_message = version_msg)

if __name__ == "__main__":
    main()
