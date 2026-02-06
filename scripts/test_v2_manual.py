import torch
from unsloth import FastLanguageModel
import json
import argparse
import os

# Configuration
# Base model matches training
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
# This points to the NEWLY TRAINED v2 adapters
V2_LORA_PATH = "outputs/cibil_qwen2.5_lora_v2"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = "You are an expert utility agent for CIBIL verification analysis. Extract the disposition and verification details from the transcript. Return ONLY valid JSON."

# Hard Mode Case 6 from samples.md (Identity Correction / Relative)
# This is a case where v1 often struggled to match the strict schema.
TRANSCRIPT = """
Agent: HI, I am Sakshi calling from HDB Financial services . Am I speaking to MINTU  KUMAR?
"""

def extract_json(text):
    """Robustly extracts JSON from model output."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        return json.loads(text[start:end])
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Manual inference for v2 model.")
    parser.add_argument("--transcript", type=str, help="Transcript text to analyze")
    args = parser.parse_args()

    transcript = args.transcript if args.transcript else TRANSCRIPT

    print(f"Loading v2 Model and Adapters...")
    print(f"Base: {BASE_MODEL}")
    print(f"LoRA: {V2_LORA_PATH}")

    if not os.path.exists(V2_LORA_PATH):
        print(f"Error: LoRA adapters not found at {V2_LORA_PATH}")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    model = FastLanguageModel.for_inference(model)
    model.load_adapter(V2_LORA_PATH)

    print(f"\n--- Analyzing Transcript ---")
    print(transcript.strip())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript.strip()}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 256,
            use_cache = True,
            do_sample = False,
        )
    
    prediction_raw = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    print("\n" + "="*50)
    print(" V2 MODEL PREDICTION")
    print("="*50)
    
    prediction = extract_json(prediction_raw)
    if prediction:
        print(json.dumps(prediction, indent=2))
    else:
        print("Model Output (Not JSON):")
        print(prediction_raw)

if __name__ == "__main__":
    main()
