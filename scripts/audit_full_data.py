import json
from collections import Counter

def audit_full_data(file_path):
    print(f"Auditing {file_path}...")
    combos = Counter()
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                analysis = json.loads(entry['messages'][2]['content'])
                
                disp = analysis.get('DISPOSITION', 'UNKNOWN')
                rpc = analysis.get('RPC_STATUS', 'UNKNOWN')
                
                key = f"{disp} | {rpc}"
                combos[key] += 1
            except:
                pass
                
    print("\n--- All Combinations Found in Source Data ---")
    for key, count in combos.most_common():
        print(f"{key:<50}: {count}")

if __name__ == "__main__":
    audit_full_data('data/processed/full_data.jsonl')
