import torch
from unsloth import FastLanguageModel
import json
import argparse
import os

# Configuration
DEFAULT_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_LORA_PATH = "khushianand01/cibil-verification-qwen2.5-lora"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = "You are an expert utility agent for CIBIL verification analysis. Extract the disposition and verification details from the transcript. Return ONLY valid JSON."

# HARDCODED TEST TRANSCRIPT
TEST_TRANSCRIPT = """
Agent: HI, I am Sakshi calling from HDB Financial services . Am I speaking to DEEPAK  BAIRWA?
User: Who is Deepika Bhairava, man? You are bothering me, calling me again and again, I have received calls three-four times. Oh man, come and check with me here. What more should I say than this? Hey, man, don't torture me, look, I received many calls yesterday as well, I got tired of replying. Or should I switch off the SIM? Hey, I don't have any loan, so... I don't have any
Agent: I understand your frustration, but according to our records, this number is linked to a RELPL loan for DEEPAK BAIRWA. Are you absolutely certain you are not DEEPAK BAIRWA?
"""

def extract_json(text):
    """Robustly extracts JSON from potentially noisy model output."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(text[start:end])
    except:
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on Cibil verification model.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--lora", type=str, default=DEFAULT_LORA_PATH, help="Path to LoRA adapters")
    parser.add_argument("--transcript", type=str, help="Override hardcoded transcript with this text")
    return parser.parse_args()

def main():
    args = parse_args()

    # Use CLI override if provided, otherwise use hardcoded
    transcript = args.transcript if args.transcript else TEST_TRANSCRIPT
    
    print(f"Loading model: {args.model} with adapters from {args.lora}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    model = FastLanguageModel.for_inference(model)
    try:
        if "huggingface.co" in args.lora or "/" in args.lora:
             # Assume it's a repo or path, try loading with PEFT for revision support if needed
             from peft import PeftModel
             # Check if it looks like our HF repo to apply v1
             revision = "v1" if "khushianand01/cibil" in args.lora else None
             model = PeftModel.from_pretrained(model, args.lora, revision=revision)
        else:
             model.load_adapter(args.lora)
    except Exception as e:
        print(f"⚠️ Error loading adapters: {e}")
        return

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
    
    prediction_raw = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    print("\nModel Output (Raw):")
    print(prediction_raw)

    # Robust JSON Validation
    prediction = extract_json(prediction_raw)
    if prediction:
        print("\n✅ Status: Success (JSON Extracted)")
        print(json.dumps(prediction, indent=2))
    else:
        print("\n❌ Status: Failed (No valid JSON found in output)")

if __name__ == "__main__":
    main()
