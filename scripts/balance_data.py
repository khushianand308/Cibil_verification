import json
import random
from collections import defaultdict
import os

def balance_data(input_file, output_file, target_cap=400, max_duplication_factor=10):
    print(f"Reading from {input_file}...")
    
    # Store data by granular bucket
    buckets = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                assistant_response = entry['messages'][2]['content']
                analysis = json.loads(assistant_response)
                
                disp = analysis.get('DISPOSITION', 'UNKNOWN')
                rpc = analysis.get('RPC_STATUS', 'UNKNOWN')
                
                # Generic Granular Buckets: Include EVERYTHING
                bucket_key = f"{disp}_{rpc}"
                    
                buckets[bucket_key].append(entry)
            except:
                continue
                
    dataset_stats = []
    balanced_data = []
    
    print("\nBalancing 8 Buckets (Target: {}, Max Dupe: {}x):".format(target_cap, max_duplication_factor))
    print(f"{'Bucket':<50} {'Original':<10} {'Balanced':<10}")
    print("-" * 75)
    
    for bucket_key, items in buckets.items():
        original_count = len(items)
        random.shuffle(items)
        
        # Calculate Effective Target
        # Limit duplication to avoided massive overfitting on rare samples
        effective_target = min(target_cap, original_count * max_duplication_factor)
        
        if original_count > effective_target:
            # Downsample to target_cap (effective_target will be target_cap here)
            selected_items = random.sample(items, effective_target)
        else:
            # Upsample to effective_target
            shortfall = effective_target - original_count
            additional_items = random.choices(items, k=shortfall)
            selected_items = items + additional_items
            
        balanced_count = len(selected_items)
        balanced_data.extend(selected_items)
        
        print(f"{bucket_key:<50} {original_count:<10} {balanced_count:<10}")

    # Mix the final dataset
    random.shuffle(balanced_data)
    
    print("-" * 75)
    print(f"Total Balanced Samples: {len(balanced_data)}")
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in balanced_data:
            f.write(json.dumps(entry) + '\n')
    print("Done.")

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(42)
    # Target 400 per bucket x 7 buckets = 2800 total samples
    balance_data('data/processed/full_data.jsonl', 'data/processed/balanced_data.jsonl', target_cap=400)
