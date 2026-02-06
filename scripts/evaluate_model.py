import torch
from unsloth import FastLanguageModel
import json
import os
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
LORA_PATH = "outputs/cibil_qwen25_lora"
TEST_FILE = "data/processed/test.jsonl"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = "You are an expert utility agent for CIBIL verification analysis. Extract the disposition and verification details from the transcript. Return ONLY valid JSON."

def extract_json(text):
    """Robustly extracts JSON from potentially noisy model output."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(text[start:end])
    except:
        return None

def count_rule_violations(row):
    """Checks for business logic inconsistencies in model predictions."""
    violations = 0
    # Rule 1: RPC Status 'full' implies both Name and Loan must be verified
    if row["pred_rpc"] == "full":
        if not row["pred_name"] or not row["pred_loan"]:
            violations += 1
    # Rule 2: Loan number verification usually requires name verification first (logical dependency)
    if row["pred_loan"] and not row["pred_name"]:
        violations += 1
    # Rule 3: Wrong number must not have identity verification set to true
    if row["pred_disp"] == "WRONG_NUMBER" and row["pred_name"]:
        violations += 1
    return violations

def main():
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    
    print(f"Loading LoRA adapters from: {LORA_PATH}")
    model = FastLanguageModel.for_inference(model)
    model.load_adapter(LORA_PATH)

    print(f"Loading test data: {TEST_FILE}")
    test_data = []
    with open(TEST_FILE, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Loaded {len(test_data)} test samples.")

    results = []

    print("Running evaluation...")
    for entry in tqdm(test_data):
        messages = entry['messages']
        user_message = messages[1]['content']
        ground_truth_str = messages[2]['content']
        ground_truth = json.loads(ground_truth_str)

        inference_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        inputs = tokenizer.apply_chat_template(
            inference_messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids = inputs,
                max_new_tokens = 256,
                use_cache = True,
                do_sample = False,
            )
        
        prediction_str = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        # Robust JSON Extraction
        prediction = extract_json(prediction_str)
        if prediction is None:
            is_valid_json = False
            prediction = {
                "DISPOSITION": "ERROR",
                "LOAN_NUMBER_VERIFIED": False,
                "NAME_VERIFIED": False,
                "RPC_STATUS": "error"
            }
        else:
            is_valid_json = True

        results.append({
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_valid_json": is_valid_json
        })

    # --- Reporting and Saving ---
    df_results = pd.DataFrame(df_entries)
    
    # Save to CSV for later inspection
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "evaluation_results_full.csv")
    df_results.to_csv(report_path, index=False)
    
    # Business Logic Checks
    df_results["rule_violations"] = df_results.apply(count_rule_violations, axis=1)
    
    # Exact Match Check (Only meaningful for valid JSON)
    df_results["exact_match"] = (
        (df_results["gt_disp"] == df_results["pred_disp"]) &
        (df_results["gt_name"] == df_results["pred_name"]) &
        (df_results["gt_loan"] == df_results["pred_loan"]) &
        (df_results["gt_rpc"] == df_results["pred_rpc"])
    )

    # Metrics for reporting
    json_validity_rate = df_results['valid_json'].mean() * 100
    valid_df = df_results[df_results["valid_json"] == True]
    exact_match_rate = valid_df["exact_match"].mean() * 100 if not valid_df.empty else 0
    total_samples = len(df_results)
    total_violations = df_results['rule_violations'].sum()
    violation_rate = df_results['rule_violations'].mean() * 100

    print("\n" + "="*50)
    print("🚀 PRODUCTION EVALUATION REPORT")
    print("="*50)
    print(f"Total Samples: {total_samples}")
    print(f"JSON Validity: {json_validity_rate:.2f}%")
    print(f"Main KPI (Exact Match Rate on valid JSON): {exact_match_rate:.2f}%")
    print(f"Rule Violations: {total_violations} ({violation_rate:.2f}%)")
    print(f"Report saved to: {report_path}")
    print("-" * 50)

    if not valid_df.empty:
        for field in ['disp', 'name', 'loan', 'rpc']:
            acc = accuracy_score(valid_df[f'gt_{field}'], valid_df[f'pred_{field}'])
            print(f"{field.upper()} Accuracy (Valid JSON only): {acc*100:.2f}%")

        print("\nDetailed Disposition Performance (Valid JSON only):")
        print(classification_report(valid_df['gt_disp'], valid_df['pred_disp'], zero_division=0))
    else:
        print("No valid JSON outputs to calculate field-level accuracy.")

if __name__ == "__main__":
    main()
