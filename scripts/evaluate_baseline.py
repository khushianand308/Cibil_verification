import json
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import argparse
import sys

# Default Model (Can be overridden)
DEFAULT_MODEL = "Qwen/Qwen3-8B"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Model Baseline")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Model path or HuggingFace ID")
    parser.add_argument("--test_file", type=str, default="data/processed/test.jsonl", help="Path to test dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick test")
    return parser.parse_args()

def extract_json(response):
    # Basic extraction: finding first { and last }
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start == -1 or end == 0: return None
        json_str = response[start:end]
        return json.loads(json_str) 
    except:
        return None

def get_chat_template(tokenizer):
    if tokenizer.chat_template is not None:
        return
    
    # Define generic ChatML template for Base models
    print("Warning: No chat template found (Base Model). Applying Qwen ChatML template.")
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    get_chat_template(tokenizer)
    
    # Load Test Data
    print(f"Loading test data: {args.test_file}")
    test_samples = []
    with open(args.test_file, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))
            
    if args.limit:
        test_samples = test_samples[:args.limit]
        print(f"Limiting to {args.limit} samples.")
        
    print(f"Evaluatin on {len(test_samples)} samples...")
    
    results = {
        "disposition_true": [], "disposition_pred": [],
        "rpc_true": [], "rpc_pred": [],
        "name_true": [], "name_pred": [],
        "loan_true": [], "loan_pred": [],
        "exact_match": 0,
        "json_error": 0
    }
    
    for entry in tqdm(test_samples):
        messages = entry['messages']
        system_prompt = messages[0]['content']
        user_prompt = messages[1]['content']
        ground_truth_str = messages[2]['content']
        ground_truth = json.loads(ground_truth_str)
        
        # Prepare Input
        # Format consistent with training
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        outputs = model.generate(prompt, max_new_tokens=512, use_cache=True, do_sample=False)
        decoded = tokenizer.batch_decode(outputs)
        response_full = decoded[0]
        
        # Extract Assistant Part (Simple heuristic or split)
        # Qwen/ChatML usually ends input with <|im_start|>assistant
        # But we can just try to parse the whole thing or look for the last JSON
        # Since we generated, the new tokens are at the end. 
        # But let's just use extract_json on the whole string, hoping user prompt doesn't have JSON.
        # Better: decode ONLY new tokens? 
        # Let's extract from <|im_start|>assistant
        try:
            response_content = response_full.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        except:
            response_content = response_full
            
        pred_json = extract_json(response_content)
        
        # Metrics
        if pred_json is None:
            results["json_error"] += 1
            # Assign dummy values for comparison
            results["disposition_pred"].append("ERROR")
            results["rpc_true"].append(str(ground_truth.get('RPC_STATUS')))
            results["rpc_pred"].append("ERROR")
            results["disposition_true"].append(ground_truth.get('DISPOSITION'))
            results["name_true"].append(ground_truth.get('NAME_VERIFIED'))
            results["name_pred"].append("ERROR")
            results["loan_true"].append(ground_truth.get('LOAN_NUMBER_VERIFIED'))
            results["loan_pred"].append("ERROR")
            continue
            
        # Normalization
        pred_disp = pred_json.get('DISPOSITION', 'MISSING')
        pred_rpc = str(pred_json.get('RPC_STATUS', 'MISSING'))
        pred_name = pred_json.get('NAME_VERIFIED', 'MISSING')
        pred_loan = pred_json.get('LOAN_NUMBER_VERIFIED', 'MISSING')
        
        true_disp = ground_truth.get('DISPOSITION')
        true_rpc = str(ground_truth.get('RPC_STATUS'))
        true_name = ground_truth.get('NAME_VERIFIED')
        true_loan = ground_truth.get('LOAN_NUMBER_VERIFIED')
        
        results["disposition_true"].append(true_disp)
        results["disposition_pred"].append(pred_disp)
        
        results["rpc_true"].append(true_rpc)
        results["rpc_pred"].append(pred_rpc)
        
        results["name_true"].append(true_name)
        results["name_pred"].append(pred_name)
        
        results["loan_true"].append(true_loan)
        results["loan_pred"].append(pred_loan)
        
        # Exact Match Check
        if (pred_disp == true_disp and pred_rpc == true_rpc and 
            pred_name == true_name and pred_loan == true_loan):
            results["exact_match"] += 1

    # Printing Report
    print(f"\n{'='*40}")
    print(f"BASELINE EVALUATION REPORT ({args.model_name})")
    print(f"{'='*40}")
    print(f"Total Samples: {len(test_samples)}")
    print(f"JSON Errors:   {results['json_error']} ({results['json_error']/len(test_samples):.1%})")
    print(f"Exact Matches: {results['exact_match']} ({results['exact_match']/len(test_samples):.1%})")
    
    print("\n--- DISPOSITION Accuracy ---")
    print(accuracy_score(results["disposition_true"], results["disposition_pred"]))
    print("\n--- RPC_STATUS Accuracy ---")
    print(accuracy_score(results["rpc_true"], results["rpc_pred"]))
    
    print("\n--- Detailed Report (Disposition) ---")
    print(classification_report(results["disposition_true"], results["disposition_pred"], zero_division=0))

if __name__ == "__main__":
    main()
