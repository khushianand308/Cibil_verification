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
    transcript: str

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
        temperature=0.1,
        max_tokens=MAX_OUTPUT_TOKENS,
        stop=["<|endoftext|>", "###", "<|im_end|>"]
    )
    
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    return final_output.outputs[0].text

def format_transcript(transcript_input):
    """Converts raw JSON or text into readable 'Role: Content' format. (From app.py)"""
    try:
        if isinstance(transcript_input, str):
            cleaned_input = transcript_input.strip()
            if cleaned_input.startswith(("{", "[")):
                data = json.loads(cleaned_input)
            else:
                return transcript_input
        else:
            data = transcript_input
            
        if isinstance(data, dict):
            transcript_list = data.get('interaction_transcript', [])
            if not transcript_list:
                 transcript_list = data.get('transcript', [])
        elif isinstance(data, list):
            transcript_list = data
        else:
            return str(transcript_input)

        dialogue = []
        for turn in transcript_list:
            if not isinstance(turn, dict): continue
            role = turn.get('role', 'unknown').capitalize()
            text = turn.get('en_text', turn.get('text', ''))
            if text: dialogue.append(f"{role}: {text}")
        
        return "\n".join(dialogue) if dialogue else str(transcript_input)
    except Exception:
        return str(transcript_input)

def clean_and_validate_analysis(data):
    """Enforces business rules and standardizes output (Matches app.py)."""
    try:
        # 1. Standardize DISPOSITION
        raw_disp = str(data.get('DISPOSITION', '')).upper().replace(" ", "_")
        
        if raw_disp == "WRONG_NUMBER": pass
        elif raw_disp == "DISCONNECTED_WITH_CONVERSATION": pass
        elif "DISCONNECTED" in raw_disp: 
            if "WITHOUT" in raw_disp: raw_disp = "DISCONNECTED_WITHOUT_CONVERSATION"
            elif "WITH" in raw_disp: raw_disp = "DISCONNECTED_WITH_CONVERSATION"
            else: raw_disp = "CALL_DROPPED"
        
        if raw_disp not in ALLOWED_DISPOSITIONS:
             raw_disp = "ANSWERED"
        
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

        # 4. Consistency Shield
        if data['DISPOSITION'] == "WRONG_NUMBER":
            data['NAME_VERIFIED'] = False
            data['LOAN_NUMBER_VERIFIED'] = False

        return {
            "disposition": data['DISPOSITION'],
            "loanNumberVerified": data['LOAN_NUMBER_VERIFIED'],
            "nameVerified": data['NAME_VERIFIED'],
            "rpcStatus": data['RPC_STATUS']
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
