import json
import os
import random

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

ALLOWED_RPC_STATUS = {"true", "false", "partial", "insufficient_data"}

def validate_logic(analysis):
    issues = []
    disp = analysis.get('DISPOSITION')
    name_ver = analysis.get('NAME_VERIFIED')
    loan_ver = analysis.get('LOAN_NUMBER_VERIFIED')
    rpc = str(analysis.get('RPC_STATUS')) # Ensure string comparison

    # Rule 1: WRONG_NUMBER implications
    if disp == "WRONG_NUMBER":
        if name_ver is not False:
            issues.append(f"WRONG_NUMBER has NAME_VERIFIED={name_ver} (Expected False)")
        # Usually loan is also not verified if wrong number
        if loan_ver is not False:
            issues.append(f"WRONG_NUMBER has LOAN_NUMBER_VERIFIED={loan_ver} (Expected False)")
        # user didn't strictly say RPC must be false, but usually it is. We won't strictly fail on RPC unless user asked, 
        # but let's check basic sanity.
    
    # Rule 2: DISCONNECTED... usually implies verified=False, unless partial?
    # We found "DISCONNECTED_WITH... | partial" exists, so name_verified CAN be True.
    # So we can't enforce name_ver=False for Disconnected.
    
    return issues

def check_quality(file_path):
    print(f"Checking quality of {file_path}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    total_rows = len(lines)
    issues_found = 0
    
    samples_by_bucket = {}
    
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            
            # Check 1: Structure
            if 'messages' not in entry:
                print(f"Row {i+1}: Missing 'messages' key")
                issues_found += 1
                continue
                
            messages = entry['messages']
            
            # Check 2: Exactly 3 roles
            if len(messages) != 3:
                print(f"Row {i+1}: Incorrect message count ({len(messages)}). Expected 3.")
                issues_found += 1
                continue
            
            if messages[0]['role'] != 'system' or messages[1]['role'] != 'user' or messages[2]['role'] != 'assistant':
                 print(f"Row {i+1}: Incorrect role order. Expected system -> user -> assistant.")
                 issues_found += 1
                 continue
            
            # Check 3: Assistant Content Parsing
            assistant_content = messages[2]['content']
            try:
                analysis = json.loads(assistant_content)
            except:
                 print(f"Row {i+1}: Assistant content is not valid JSON")
                 issues_found += 1
                 continue

            # Check 4: Schema & Enum
            disp = analysis.get('DISPOSITION')
            if disp not in ALLOWED_DISPOSITIONS:
                print(f"Row {i+1}: Invalid DISPOSITION '{disp}'")
                issues_found += 1
                
            rpc = str(analysis.get('RPC_STATUS'))
            if rpc not in ALLOWED_RPC_STATUS:
                 print(f"Row {i+1}: Invalid RPC_STATUS '{rpc}'")
                 issues_found += 1
                 
            # Check 5: Logical Rules
            logic_issues = validate_logic(analysis)
            if logic_issues:
                for issue in logic_issues:
                    print(f"Row {i+1} Logic Error: {issue}")
                issues_found += 1

            # Sampling (Valid rows only)
            if issues_found == 0: 
                bucket_key = f"{disp} | {rpc}"
                if bucket_key not in samples_by_bucket:
                    samples_by_bucket[bucket_key] = {
                        "system_prompt": messages[0]['content'],
                        "transcript": messages[1]['content'],
                        "analysis": analysis
                    }

        except json.JSONDecodeError:
            print(f"Row {i+1}: Invalid JSON line")
            issues_found += 1
            
    print(f"Total Rows: {total_rows}")
    print(f"Issues: {issues_found}")
    return samples_by_bucket

if __name__ == "__main__":
    files = [
        'data/processed/train.jsonl',
        'data/processed/val.jsonl',
        'data/processed/test.jsonl'
    ]
    
    all_samples = {}
    
    for fp in files:
        if os.path.exists(fp):
            print(f"\n[{fp}]")
            samples = check_quality(fp)
            if "train" in fp: all_samples.update(samples)
        else:
            print(f"File not found: {fp}")

    print("\n--- Random Samples (from Train) ---")
    for bucket, sample in all_samples.items():
        print(f"\nCategory: [{bucket}]")
        print(f"System: {sample['system_prompt']}")
        print(f"Transcript (Truncated): {sample['transcript'][:100]}...")
        print(f"Label: {json.dumps(sample['analysis'])}")
