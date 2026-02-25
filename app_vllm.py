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

def format_transcript_from_list(lines: List[TranscriptLine]) -> str:
    formatted = []
    for line in lines:
        role = "Agent" if line.role.lower() == "agent" else "User"
        formatted.append(f"{role}: {line.en_text}")
    return "\n".join(formatted)

def clean_and_validate_analysis(data):
    """Enforces business rules and standardizes output."""
    try:
        # 1. Standardize DISPOSITION
        disposition = data.get("disposition") or data.get("Disposition") or "ANSWERED"
        raw_disp = str(disposition).upper().replace(" ", "_").strip()
        
        # Mapping for vLLM specific variants
        if "WRONG" in raw_disp: raw_disp = "WRONG_NUMBER"
        elif "DISCONNECTED" in raw_disp: 
            if "WITHOUT" in raw_disp: raw_disp = "DISCONNECTED_WITHOUT_CONVERSATION"
            else: raw_disp = "DISCONNECTED_WITH_CONVERSATION"
        
        # Strict Check fallback
        if raw_disp not in ALLOWED_DISPOSITIONS:
             raw_disp = "ANSWERED"
        
        # 2. Standardize RPC_STATUS
        rpc_s = str(data.get("rpcStatus") or "insufficient_data").lower().strip()
        if "complete" in rpc_s or "true" in rpc_s: rpc_s = "true"
        elif "false" in rpc_s: rpc_s = "false"
        elif rpc_s not in ALLOWED_RPC_STATUS: rpc_s = "insufficient_data"

        # 3. Clean Booleans & Consistency
        name_v = bool(data.get("nameVerified"))
        loan_v = bool(data.get("loanNumberVerified"))

        if raw_disp == "WRONG_NUMBER":
            name_v = False
            loan_v = False

        return {
            "disposition": raw_disp,
            "loanNumberVerified": loan_v,
            "nameVerified": name_v,
            "rpcStatus": rpc_s
        }
    except Exception:
        return data

def parse_llm_json(raw_text: str, transcript_text: str = ""):
    try:
        clean_text = raw_text.strip()
        start = clean_text.find('{')
        end = clean_text.rfind('}') + 1
        data_raw = {}
        
        if start != -1 and end != 0:
            json_str = clean_text[start:end]
            data_raw = json.loads(json_str)
        
        # --- Contextual Safety Net ---
        # If JSON is empty or missing keys, look for obvious patterns in the transcript
        if not data_raw or not any(data_raw.values()):
            transcript_upper = transcript_text.upper()
            if "WRONG NUMBER" in transcript_upper or "NOT PAWAN" in transcript_upper:
                return clean_and_validate_analysis({"disposition": "WRONG_NUMBER"})
            if "DISCONNECTED" in transcript_upper or "CALL DROPPED" in transcript_upper:
                return clean_and_validate_analysis({"disposition": "DISCONNECTED_WITH_CONVERSATION"})

        # Extract nested or flat keys
        v_details_raw = data_raw.get("Verification_Details", {})
        v_details = {}
        if isinstance(v_details_raw, dict):
            v_details = v_details_raw
        elif isinstance(v_details_raw, list) and len(v_details_raw) > 0:
            # Handle list of dicts or standard list fallback
            v_details = v_details_raw[0] if isinstance(v_details_raw[0], dict) else {}

        intermediate = {
            "disposition": data_raw.get("DISPOSITION") or data_raw.get("Disposition"),
            "nameVerified": data_raw.get("NAME_VERIFIED") or v_details.get("Name") or v_details.get("Name_Verified"),
            "loanNumberVerified": data_raw.get("LOAN_NUMBER_VERIFIED") or v_details.get("Loan_Number") or v_details.get("Loan_Number_Verified"),
            "rpcStatus": data_raw.get("RPC_STATUS") or v_details.get("Call_Status") or v_details.get("RPC_Status")
        }
        
        return clean_and_validate_analysis(intermediate)
    except Exception:
        pass
    return {"error": "Failed to parse JSON", "raw_output": raw_text}

# --- Endpoints ---
@app.post("/verify")
async def verify(request: VerificationRequest):
    prediction_raw = await generate_vllm(request.transcript)
    print(f"DEBUG - Raw Prediction: {prediction_raw}")
    return parse_llm_json(prediction_raw, request.transcript)

@app.post("/verify-batch")
async def verify_batch(request: BatchRequest):
    results = []
    
    # Process batch using asyncio.gather for true parallel vLLM execution
    async def process_item(item):
        transcript_text = format_transcript_from_list(item.interaction_transcript)
        prediction_raw = await generate_vllm(transcript_text)
        return parse_llm_json(prediction_raw, transcript_text)

    results = await asyncio.gather(*[process_item(item) for item in request.transcripts])
    return results

if __name__ == "__main__":
    import uvicorn
    # vLLM is native async, we only need 1 worker!
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)
