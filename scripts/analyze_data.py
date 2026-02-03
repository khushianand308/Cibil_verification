import json
from collections import Counter

def analyze_distribution(file_path):
    dispositions = []
    rpc_statuses = []
    name_verified = []
    loan_verified = []
    
    total = 0
    errors = 0

    print(f"Analyzing {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            total += 1
            try:
                entry = json.loads(line)
                assistant_response = entry['messages'][2]['content']
                analysis = json.loads(assistant_response)
                
                dispositions.append(analysis.get('DISPOSITION', 'UNKNOWN'))
                rpc_statuses.append(analysis.get('RPC_STATUS', 'UNKNOWN'))
                name_verified.append(str(analysis.get('NAME_VERIFIED', 'UNKNOWN')))
                loan_verified.append(str(analysis.get('LOAN_NUMBER_VERIFIED', 'UNKNOWN')))
            except Exception as e:
                errors += 1
                continue

    print(f"\nTotal Samples: {total}")
    print(f"Errors: {errors}")
    
    def print_dist(name, data):
        counts = Counter(data)
        print(f"\n{name} Distribution:")
        for label, count in counts.most_common():
            percentage = (count / total) * 100
            print(f"  {label:<35}: {count} ({percentage:.2f}%)")

    print_dist("DISPOSITION", dispositions)
    print_dist("RPC_STATUS", rpc_statuses)
    print_dist("NAME_VERIFIED", name_verified)
    print_dist("LOAN_NUMBER_VERIFIED", loan_verified)

if __name__ == "__main__":
    analyze_distribution('data/processed/balanced_data.jsonl')
