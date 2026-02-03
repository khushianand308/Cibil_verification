import pandas as pd
import json

def scan_raw_values():
    try:
        df = pd.read_csv('data/raw_data.csv')
        
        dispositions = set()
        rpc_statuses = set()
        name_ver = set()
        loan_ver = set()
        
        print("Scanning raw CSV...")
        for val in df['post_call_analysis']:
            try:
                data = json.loads(val)
                dispositions.add(data.get('DISPOSITION'))
                rpc_statuses.add(data.get('RPC_STATUS'))
                name_ver.add(data.get('NAME_VERIFIED'))
                loan_ver.add(data.get('LOAN_NUMBER_VERIFIED'))
            except:
                pass
                
        print("\n--- Unique Values Found ---")
        print("DISPOSITIONS:", sorted([str(x) for x in dispositions if x is not None]))
        print("RPC_STATUS:", sorted([str(x) for x in rpc_statuses if x is not None]))
        print("NAME_VERIFIED:", sorted([str(x) for x in name_ver if x is not None]))
        print("LOAN_NUMBER_VERIFIED:", sorted([str(x) for x in loan_ver if x is not None]))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    scan_raw_values()
