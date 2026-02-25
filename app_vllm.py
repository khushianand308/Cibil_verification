import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, Any, List
import json
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

# Initialize FastAPI App
app = FastAPI(title="Cibil Verification API (vLLM)", version="3.0")

# --- Configuration ---
# Point to the MERGED model folder
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/cibil-qwen2.5-7b-v2-merged")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "128"))

# Initialize vLLM Engine
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    gpu_memory_utilization=0.90, # Safer for sampler warmup on T4
    max_model_len=MAX_SEQ_LENGTH,
    trust_remote_code=True,
    # T4 specific optimization
    enforce_eager=True, 
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Load Tokenizer once globally
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)

# Request/Response Schemas (Same as v2)
# Allowed Values Constants (Matches production v2)
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

class VerificationRequest(BaseModel):
    transcript: Union[str, dict, list, Any]

class TranscriptLine(BaseModel):
    role: str
    en_text: str

class BatchTranscriptItem(BaseModel):
    interaction_transcript: List[TranscriptLine]

class BatchRequest(BaseModel):
    transcripts: List[BatchTranscriptItem]

# --- Helper Functions ---
async def generate_vllm(transcript_text: str):
    # Apply Chat Template with System Prompt (Matches production vllm/v2)
    messages = [
        {"role": "system", "content": "You are an expert utility agent for CIBIL verification analysis. Extract the disposition and verification details from the transcript. Return ONLY valid JSON."},
        {"role": "user", "content": transcript_text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"DEBUG - Full Prompt: {prompt}")

    request_id = random_uuid()
    sampling_params = SamplingParams(
        temperature=0.0, # Greedy (Matches app.py do_sample=False)
        max_tokens=MAX_OUTPUT_TOKENS,
        stop=["<|endoftext|>", "###", "<|im_end|>"]
    )
    
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    return final_output.outputs[0].text

def format_transcript(transcript_input):
    """Robust transcript formatter that handles dict, list, and raw string."""
    try:
        if isinstance(transcript_input, str):
            cleaned_input = transcript_input.strip()
            if cleaned_input.startswith(("{", "[")):
                data = json.loads(cleaned_input)
            else:
                return transcript_input
        else:
            data = transcript_input
            
        # Extract interaction_transcript if nested
        if isinstance(data, dict):
            transcript_list = data.get('interaction_transcript') or data.get('transcript') or []
        elif isinstance(data, list):
            transcript_list = data
        else:
            return str(transcript_input)

        dialogue = []
        for turn in transcript_list:
            if not isinstance(turn, dict): continue
            # Handle variations in role/text keys
            role = (turn.get('role') or "unknown").capitalize()
            text = turn.get('en_text') or turn.get('text') or ""
            if text: dialogue.append(f"{role}: {text}")
        
        return "\n".join(dialogue) if dialogue else str(transcript_input)
    except Exception:
        return str(transcript_input)

def clean_and_validate_analysis(data):
    """Hyper-robust logic to standardize model output regardless of training drift."""
    try:
        # Helper for case-insensitive lookup
        def get_val(d, *keys):
            for k in keys:
                # Direct check
                if k in d: return d[k]
                # Case-insensitive check
                for dk in d.keys():
                    if dk.lower() == k.lower(): return d[dk]
            return None

        # 1. Standardize DISPOSITION
        raw_disp = str(get_val(data, 'DISPOSITION', 'disposition') or "").upper().replace(" ", "_").strip()
        
        if "WRONG" in raw_disp: raw_disp = "WRONG_NUMBER"
        elif "DISCONNECTED" in raw_disp: 
            if "WITHOUT" in raw_disp: raw_disp = "DISCONNECTED_WITHOUT_CONVERSATION"
            else: raw_disp = "DISCONNECTED_WITH_CONVERSATION"
        
        if raw_disp not in ALLOWED_DISPOSITIONS:
             raw_disp = "ANSWERED"
        
        # 2. Standardize RPC_STATUS
        raw_rpc = str(get_val(data, 'RPC_STATUS', 'rpcStatus', 'rpc_status') or "").lower().strip()
        if "true" in raw_rpc: res_rpc = "true"
        elif "false" in raw_rpc: res_rpc = "false"
        elif raw_rpc in ALLOWED_RPC_STATUS: res_rpc = raw_rpc
        else: res_rpc = "insufficient_data"

        # 3. Handle Nested Verification_Details if model emits them (found in some merged versions)
        v_details = get_val(data, 'Verification_Details', 'verification_details') or {}
        if not isinstance(v_details, dict): v_details = {}
        
        # 4. Clean Booleans
        name_v = get_val(data, 'NAME_VERIFIED', 'nameVerified') or get_val(v_details, 'Name_Verified', 'Customer_Name', 'Name')
        loan_v = get_val(data, 'LOAN_NUMBER_VERIFIED', 'loanNumberVerified') or get_val(v_details, 'Loan_Number_Verified', 'Loan_Number_Last_Four_Digits', 'Loan_ID')

        name_v = bool(name_v)
        loan_v = bool(loan_v)

        # 5. Consistency Shield
        if raw_disp == "WRONG_NUMBER":
            name_v = False
            loan_v = False

        return {
            "disposition": raw_disp,
            "loanNumberVerified": loan_v,
            "nameVerified": name_v,
            "rpcStatus": res_rpc
        }
    except Exception:
        return data

def parse_llm_json(raw_text: str):
    """Robustly extracts and validates JSON from model output."""
    try:
        clean_text = raw_text.strip()
        start = clean_text.find('{')
        end = clean_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = clean_text[start:end]
            data_raw = json.loads(json_str)
            return clean_and_validate_analysis(data_raw)
    except Exception:
        pass
    return {"error": "Failed to parse JSON", "raw_output": raw_text}

# --- Endpoints ---
@app.post("/verify")
async def verify(request: VerificationRequest):
    processed_transcript = format_transcript(request.transcript)
    prediction_raw = await generate_vllm(processed_transcript)
    print(f"DEBUG - Raw Prediction: {prediction_raw}")
    return parse_llm_json(prediction_raw)

@app.post("/verify-batch")
async def verify_batch(request: BatchRequest):
    results = []
    
    async def process_item(item):
        processed_transcript = format_transcript(item) # format_transcript handles dict/BatchTranscriptItem
        prediction_raw = await generate_vllm(processed_transcript)
        return parse_llm_json(prediction_raw)

    results = await asyncio.gather(*[process_item(item) for item in request.transcripts])
    return results

if __name__ == "__main__":
    import uvicorn
    # vLLM is native async, we only need 1 worker!
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)
