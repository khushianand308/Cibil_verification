# Cibil Verification Model Fine-Tuning (v2 Production)

This project contains the pipeline and scripts for fine-tuning a Small Language Model (SLM) to extract structured verification details from call transcripts. 

## 🎯 Project Goal
The objective is to analyze customer call transcripts and output a structured JSON containing:
1. `DISPOSITION`: The final status of the call.
2. `NAME_VERIFIED`: Boolean flag if the customer name was confirmed.
3. `LOAN_NUMBER_VERIFIED`: Boolean flag if the loan details were confirmed.
4. `RPC_STATUS`: Right Party Contact status (true, false, partial, insufficient_data).

---

## 🚀 Performance Snapshot (v2 Production)
The model was fine-tuned on **12,000 balanced samples** using `Qwen2.5-7B-Instruct-bnb-4bit`.

| Metric | v2 Score | Status |
| :--- | :--- | :--- |
| **Overall Accuracy (Exact Match)** | **89.94%** | PASS |
| **JSON Validity Rate** | **100.00%** | PASS |
| **Rule Violation Rate** | **0.97%** | PASS |

### Field-Level Metrics (v2):
- **Loan Number Verification**: 98.15%
- **Name Verification**: 97.62%
- **Disposition Accuracy**: 88.79%
- **RPC Status Accuracy**: 89.94%

---

## 🛠️ Environment Setup
The project uses the `cibil_disposition` conda environment.

```bash
# 1. Activate Environment
conda activate cibil_disposition

# 2. Install Dependencies
pip install -r requirements.txt
```

---

## 🏃 Execution Flow

### 1. Data Preparation
Processes raw data into balanced 12k datasets.
```bash
conda activate cibil_disposition
python scripts/preprocess_data.py
python scripts/balance_data.py
python scripts/split_data.py
```

### 2. Fine-Tuning (v2)
Fine-tune the model on the GPU.
```bash
conda activate cibil_disposition
python scripts/train.py
```
*Outputs save to `outputs/cibil_qwen2.5_lora_v2/`.*

### 3. Evaluation
Run the full production evaluation on the test set.
```bash
conda activate cibil_disposition
python scripts/evaluate_v2.py
```

### 4. Deployment & Export
Merge the LoRA weights and push to Hugging Face.
```bash
conda activate cibil_disposition
python scripts/export_model.py
```

---



---

## 🧪 Testing with CURL

Use these commands to verify the API logic.

### Case 1: Full Success
```bash
curl -X POST "http://localhost:9090/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: HI, I am Sakshi calling from HDB Financial services . Am I speaking to SHAIK MUBEENA?\nUser: Yes.\nAgent: The loan account number ends with 97. Can you confirm if this loan belongs to you?\nUser: Yes, it is mine."}'
```

### Case 2: Wrong Number
```bash
curl -X POST "http://localhost:9090/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: Am I speaking to DEEPAK DWIVEDI?\nUser: No, wrong number."}'
```

### Case 3: Identity Correction (Relative speaking)
```bash
curl -X POST "http://localhost:9090/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: Am I speaking to RISHAL KAIBARTA?\nUser: No, I am his brother."}'
```

---

## 📂 Directory Structure
- `scripts/`: Implementation scripts (v2 specific).
- `data/processed/`: Balanced 12k training/test datasets.
- `models/`: Merged standalone 15GB models (local only).
- `requirements.txt`: Project dependencies.
- `reports/`: Evaluation summaries.

## 📝 License
Proprietary / Internal Use Only.
