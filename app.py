import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
import json

# Initialize FastAPI App
app = FastAPI(title="Cibil Verification API", version="1.0")

# --- Configuration ---
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTERS_REPO = "khushianand01/cibil-verification-qwen2.5-lora" # Your Hugging Face Repo
MAX_SEQ_LENGTH = 2048
dtype = None # Auto detection
load_in_4bit = True

# Global Model & Tokenizer
model = None
tokenizer = None

class VerificationRequest(BaseModel):
    transcript: str

def extract_json(text):
    """Robustly extracts JSON from model output."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(text[start:end])
    except:
        return None

@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    global model, tokenizer
    print(" Loading Model... This may take a minute.")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # Load and activate adapters from your HF repo
    print(f"🔗 Loading Adapters from {ADAPTERS_REPO}...")
    
    # Correction: We do NOT use get_peft_model here because that initializes NEW random weights for training.
    # Instead, we just load the saved adapter we trained.
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTERS_REPO, revision="v1")
    
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    print("✅ Model Loaded and Ready!")

@app.post("/verify")
async def verify_transcript(request: VerificationRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    messages = [
        {"role": "system", "content": "You are an expert utility agent for CIBIL verification analysis. Extract the disposition and verification details from the transcript. Return ONLY valid JSON."},
        {"role": "user", "content": request.transcript}
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
    
    parsed_json = extract_json(prediction_raw)
    
    if parsed_json:
        return parsed_json
    else:
        return {"error": "Failed to parse JSON", "raw_output": prediction_raw}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
