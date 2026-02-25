import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Union, Any, List
from unsloth import FastLanguageModel
import json

# Initialize FastAPI App
app = FastAPI(title="Cibil Verification API", version="2.0")

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
ADAPTERS_REPO = os.getenv("ADAPTERS_REPO", "khushianand01/cibil-verification-qwen2.5-lora")
ADAPTERS_REVISION = os.getenv("ADAPTERS_REVISION", "v2")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
dtype = None # Auto detection
load_in_4bit = os.getenv("LOAD_IN_4BIT", "True").lower() == "true"

# Allowed Values Constants
ALLOWED_DISPOSITIONS = {
    "ANSWERED", 
    "DISCONNECTED_WITHOUT_CONVERSATION", 
    "DISCONNECTED_WITH_CONVERSATION",
    "WRONG_NUMBER", 
}

ALLOWED_RPC_STATUS = {
    "true", 
    "false", 
    "partial", 
    "insufficient_data"
}

# Global Model & Tokenizer
model = None
tokenizer = None

class VerificationRequest(BaseModel):
    transcript: Union[str, dict, list, Any]

class BatchVerificationRequest(BaseModel):
    transcripts: List[Union[str, dict, list, Any]]

def format_transcript(transcript_input):
    """Converts raw JSON or text into readable 'Role: Content' format."""
    try:
        # 1. Parse string to JSON if needed
        if isinstance(transcript_input, str):
            cleaned_input = transcript_input.strip()
            if cleaned_input.startswith(("{", "[")):
                data = json.loads(cleaned_input)
            else:
                return transcript_input # Return raw text
        else:
            data = transcript_input
            
        # 2. Extract transcript list
        if isinstance(data, dict):
            # Check for specific 'interaction_transcript' key
            transcript_list = data.get('interaction_transcript', [])
            if not transcript_list and not any(k in data for k in ['interaction_transcript']):
                 # Fallback for other standard dict formats
                 transcript_list = data.get('transcript', [])
        elif isinstance(data, list):
            transcript_list = data
        else:
            return str(transcript_input)

        # 3. Build dialogue
        dialogue = []
        for turn in transcript_list:
            if not isinstance(turn, dict):
                continue
            role = turn.get('role', 'unknown').capitalize()
            # Support both 'en_text' (new format) and 'text' (old format)
            text = turn.get('en_text', turn.get('text', ''))
            if text:
                dialogue.append(f"{role}: {text}")
        
        return "\n".join(dialogue) if dialogue else str(transcript_input)
    except Exception as e:
        print(f"Transcript formatting error: {e}")
        return str(transcript_input)

def clean_and_validate_analysis(data):
    """Enforces business rules and standardizes output (Dict version of preprocess logic)."""
    try:
        # 1. Standardize DISPOSITION
        raw_disp = str(data.get('DISPOSITION', '')).upper().replace(" ", "_")
        
        # Mapping logic from preprocess_data.py
        if raw_disp == "WRONG_NUMBER": pass
        elif raw_disp == "DISCONNECTED_WITH_CONVERSATION": pass
        elif "DISCONNECTED" in raw_disp: 
            if "WITHOUT" in raw_disp: raw_disp = "DISCONNECTED_WITHOUT_CONVERSATION"
            elif "WITH" in raw_disp: raw_disp = "DISCONNECTED_WITH_CONVERSATION"
            else: raw_disp = "CALL_DROPPED"
        
        # Strict Check fallback
        if raw_disp not in ALLOWED_DISPOSITIONS:
             raw_disp = "ANSWERED" # Default for API rather than dropping
        
        data['DISPOSITION'] = raw_disp

        # 2. Standardize RPC_STATUS
        raw_rpc = str(data.get('RPC_STATUS', '')).lower()
        if raw_rpc == "true": data['RPC_STATUS'] = "true"
        elif raw_rpc == "false": data['RPC_STATUS'] = "false"
        elif raw_rpc in ALLOWED_RPC_STATUS:
            data['RPC_STATUS'] = raw_rpc
        else:
            data['RPC_STATUS'] = "insufficient_data"

        # 3. Clean Booleans
        data['LOAN_NUMBER_VERIFIED'] = bool(data.get('LOAN_NUMBER_VERIFIED', False))
        data['NAME_VERIFIED'] = bool(data.get('NAME_VERIFIED', False))

        # 4. Consistency Shield: WRONG_NUMBER -> NAME/LOAN must be False
        if data['DISPOSITION'] == "WRONG_NUMBER":
            data['NAME_VERIFIED'] = False
            data['LOAN_NUMBER_VERIFIED'] = False

        # 5. Build cleaned response (camelCase for Production)
        return {
            "disposition": data['DISPOSITION'],
            "loanNumberVerified": data['LOAN_NUMBER_VERIFIED'],
            "nameVerified": data['NAME_VERIFIED'],
            "rpcStatus": data['RPC_STATUS']
        }
    except Exception as e:
        print(f"Validation error: {e}")
        return data

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
    """Load model and production V2 adapters on server startup."""
    global model, tokenizer
    print("Loading Model... This may take a minute.")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # Load and activate adapters from your HF repo
    print(f"Loading Adapters from {ADAPTERS_REPO} (branch: {ADAPTERS_REVISION})...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTERS_REPO, revision=ADAPTERS_REVISION)
    
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    print(f"{ADAPTERS_REVISION} Production API Ready!")

@app.post("/verify")
async def verify_transcript(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    
    # Fully backwards compatible: Check if payload holds exactly one 'transcript' key
    if isinstance(data, dict) and "transcript" in data and len(data) == 1:
        transcript_data = data["transcript"]
    else:
        transcript_data = data
        
    return await process_single_transcript(transcript_data)

@app.post("/verify-batch")
async def verify_batch(request: BatchVerificationRequest):
    """Processes a list of transcripts sequentially."""
    results = []
    for transcript_data in request.transcripts:
        result = await process_single_transcript(transcript_data)
        results.append(result)
    return results

async def process_single_transcript(transcript_input):
    """Helper to process a single transcript through the model."""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Apply Transcript Preprocessing (Cleaning)
    processed_transcript = format_transcript(transcript_input)

    messages = [
        {"role": "system", "content": "You are an expert utility agent for CIBIL verification analysis. Extract the disposition and verification details from the transcript. Return ONLY valid JSON."},
        {"role": "user", "content": processed_transcript}
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
        # Apply Logic Guardians (Post-processing)
        final_json = clean_and_validate_analysis(parsed_json)
        return final_json
    else:
        return {"error": "Failed to parse JSON", "raw_output": prediction_raw}

if __name__ == "__main__":
    # For 16GB VRAM, 1 heavy worker is safest for queueing stability.
    workers = int(os.getenv("WORKERS", "1"))
    
    print(f"Starting server on port 5000 with {workers} worker(s)...")
    print("Queueing enabled: Up to 30 concurrent connections supported.")
    
    # backlog=2048: Allows a large queue of waiting connections.
    # timeout_keep_alive=120: Prevents disconnects during long generations.
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=5000, 
        workers=workers,
        backlog=2048,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )
