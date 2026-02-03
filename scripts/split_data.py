import json
import random
import os
from collections import defaultdict

def split_data(input_file, output_dir, val_size=0.1, test_size=0.1):
    print(f"Reading from {input_file}...")
    
    # Bucket data to ensure stratified split
    buckets = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                analysis = json.loads(entry['messages'][2]['content'])
                
                disp = analysis.get('DISPOSITION', 'UNKNOWN')
                rpc = analysis.get('RPC_STATUS', 'UNKNOWN')
                # Strict normalization
                bucket_key = f"{disp}_{rpc}"
                
                buckets[bucket_key].append(entry)
            except:
                continue
    
    train_data = []
    val_data = []
    test_data = []
    
    print(f"\nSplitting data (Val: {val_size}, Test: {test_size})...")
    print(f"{'Bucket':<50} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 90)
    
    for bucket_key, items in buckets.items():
        random.shuffle(items)
        n_total = len(items)
        
        n_test = max(1, int(n_total * test_size))
        n_val = max(1, int(n_total * val_size))
        
        # Ensure we don't over-allocate if data is tiny
        if n_test + n_val >= n_total:
            # Fallback for tiny buckets (like 1-2 items)
            n_test = 1
            n_val = 0 if n_total < 2 else 1
            n_train = max(0, n_total - n_test - n_val)
        else:
            n_train = n_total - n_test - n_val
        
        test_items = items[:n_test]
        val_items = items[n_test:n_test+n_val]
        train_items = items[n_test+n_val:]
        
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)
        
        print(f"{bucket_key:<50} {n_total:<8} {n_train:<8} {n_val:<8} {n_test:<8}")
        
    # Shuffle final sets
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print("-" * 90)
    print(f"Total Train: {len(train_data)}")
    print(f"Total Val:   {len(val_data)}")
    print(f"Total Test:  {len(test_data)}")
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.jsonl')
    val_path = os.path.join(output_dir, 'val.jsonl')
    test_path = os.path.join(output_dir, 'test.jsonl')
    
    with open(train_path, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
            
    with open(val_path, 'w') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')
            
    with open(test_path, 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved to:\n  - {train_path}\n  - {val_path}\n  - {test_path}")

if __name__ == "__main__":
    split_data('data/processed/balanced_data.jsonl', 'data/processed', val_size=0.1, test_size=0.1)
