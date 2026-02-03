import pandas as pd
import json
import os

def format_transcript(transcript_input):
    """
    Converts the transcript JSON list into a readable dialogue string.
    Handles nested structure: {"interaction_transcript": [...]}
    """
    try:
        # Parse if string
        if isinstance(transcript_input, str):
            data = json.loads(transcript_input)
        else:
            data = transcript_input
            
        # Extract the list if it's inside a dictionary
        if isinstance(data, dict):
            transcript_list = data.get('interaction_transcript', [])
        elif isinstance(data, list):
            transcript_list = data
        else:
            return None
            
        # Check if list is valid
        if not isinstance(transcript_list, list):
            return None
        
        # Build dialogue
        dialogue = []
        for turn in transcript_list:
            if not isinstance(turn, dict):
                continue
            role = turn.get('role', 'unknown').capitalize()
            text = turn.get('en_text', '')
            dialogue.append(f"{role}: {text}")
        
        return "\n".join(dialogue)
    except Exception as e:
        # print(f"Error formatting transcript: {e}") # Reduce noise
        return None

# Allowed Values Constants
ALLOWED_DISPOSITIONS = {
    "ANSWERED", 
    "DISCONNECTED_WITHOUT_CONVERSATION", 
    "DISCONNECTED_WITH_CONVERSATION",
    "CALL_DROPPED", 
    "WRONG_NUMBER", 
    "REFUSED", 
    "VERIFIED", 
    "NOT_VERIFIED"
}

ALLOWED_RPC_STATUS = {
    "true", 
    "false", 
    "partial", 
    "insufficient_data"
}

def clean_and_validate_analysis(analysis_str):
    """
    Parses, validates, and standardizes the analysis JSON.
    Returns cleaned JSON string or None if invalid.
    """
    try:
        data = json.loads(analysis_str)
        
        # 1. Standardize DISPOSITION
        raw_disp = data.get('DISPOSITION', '').upper().replace(" ", "_")
        # specific mappings for common mismatches in raw data
        if raw_disp == "WRONG_NUMBER": raw_disp = "WRONG_NUMBER" # matches
        # Fix specific raw data casing or mapping issues
        elif raw_disp == "DISCONNECTED_WITH_CONVERSATION": pass # Keep as is
        elif "DISCONNECTED" in raw_disp: 
            # Default mapping for variations
            if "WITHOUT" in raw_disp: raw_disp = "DISCONNECTED_WITHOUT_CONVERSATION"
            elif "WITH" in raw_disp: raw_disp = "DISCONNECTED_WITH_CONVERSATION"
            else: raw_disp = "CALL_DROPPED" # Fallback
        
        # Strict Check
        if raw_disp not in ALLOWED_DISPOSITIONS:
            # Try to map 'Wrong Number' -> 'WRONG_NUMBER' if not caught above
             if raw_disp == "WRONG_NUMBER": pass
             else: return None # Discard unrelated values
        
        data['DISPOSITION'] = raw_disp

        # 2. Standardize RPC_STATUS
        # User requested: true, false, partial, insufficient_data
        raw_rpc = str(data.get('RPC_STATUS', '')).lower()
        
        # Convert booleans to string 'true'/'false' if they aren't already
        if raw_rpc == "true": data['RPC_STATUS'] = "true"
        elif raw_rpc == "false": data['RPC_STATUS'] = "false"
        elif raw_rpc in ALLOWED_RPC_STATUS:
            data['RPC_STATUS'] = raw_rpc
        else:
            # Default fallback if unknown
            data['RPC_STATUS'] = "insufficient_data"

        # 3. Type Checks for Booleans
        data['LOAN_NUMBER_VERIFIED'] = bool(data.get('LOAN_NUMBER_VERIFIED', False))
        data['NAME_VERIFIED'] = bool(data.get('NAME_VERIFIED', False))
        
        # 4. Logical Consistency Enforcement (User Request)
        # Rule: WRONG_NUMBER -> NAME_VERIFIED=False, LOAN_NUMBER_VERIFIED=False
        if data['DISPOSITION'] == "WRONG_NUMBER":
            data['NAME_VERIFIED'] = False
            data['LOAN_NUMBER_VERIFIED'] = False

        # 5. Construct Final Ordered Dictionary to ensure schema compliance
        clean_data = {
            "DISPOSITION": data['DISPOSITION'],
            "LOAN_NUMBER_VERIFIED": data['LOAN_NUMBER_VERIFIED'],
            "NAME_VERIFIED": data['NAME_VERIFIED'],
            "RPC_STATUS": data['RPC_STATUS']
        }
        
        return json.dumps(clean_data)
    except:
        return None

def create_chatml_entry(row):
    """
    Creates a single ChatML training entry.
    """
    try:
        transcript_str = format_transcript(row['interaction_transcript'])
        if not transcript_str:
            return None

        # Clean/Parse the analysis JSON
        analysis_str = clean_and_validate_analysis(row['post_call_analysis'])
        if not analysis_str:
            return None
        
        # Construct messages
        messages = [
            {"role": "system", "content": "You are an expert utility agent for CIBIL verification analysis. Extract the disposition and verification details from the transcript. Return ONLY valid JSON."},
            {"role": "user", "content": transcript_str},
            {"role": "assistant", "content": analysis_str} # Keeping as JSON string output
        ]
        
        return {"messages": messages}
    except Exception as e:
        # print(f"Error creating entry for row: {e}")
        return None

def main():
    input_file = 'data/raw_data.csv'
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")

    # Drop rows with missing essential data
    df = df.dropna(subset=['interaction_transcript', 'post_call_analysis'])
    print(f"Rows after dropping nulls: {len(df)}")

    # Apply processing
    print("Processing rows...")
    processed_data = []
    
    # Iterate and process
    for idx, row in df.iterrows():
        entry = create_chatml_entry(row)
        if entry:
            processed_data.append(entry)
            
    print(f"Successfully processed {len(processed_data)} samples.")
    
    if len(processed_data) == 0:
        print("Error: No samples processed. Exiting.")
        return

    # Save to JSONL (No split)
    train_path = os.path.join(output_dir, 'full_data.jsonl')
    
    with open(train_path, 'w') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved {len(processed_data)} samples to {train_path}")

if __name__ == "__main__":
    main()
