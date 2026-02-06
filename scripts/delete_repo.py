from huggingface_hub import HfApi
import sys

# Configuration
REPO_ID = "khushianand01/cibil-verification-v2-lora"

def main():
    api = HfApi()
    print(f"Attempting to delete repository: {REPO_ID}...")
    try:
        api.delete_repo(repo_id=REPO_ID, repo_type="model")
        print(f"Successfully deleted repository: {REPO_ID}")
    except Exception as e:
        print(f"Error deleting repository: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
