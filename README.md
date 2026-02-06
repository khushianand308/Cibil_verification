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

| Metric | v2 Score | status |
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
The project uses the `disposition_v3` conda environment.

```bash
conda activate disposition_v3
pip install -r requirements.txt
```

**Key Dependencies**:
- `unsloth`: For 2x faster, 4-bit memory-efficient fine-tuning.
- `transformers`, `peft`, `trl`: Hugging Face training stack.
- `fastapi`, `uvicorn`: For the Production API.

---

## 🏃 Execution Flow

### 1. Data Preparation
Processes raw data into balanced 12k datasets.
```bash
python scripts/preprocess_data.py
python scripts/balance_data.py
python scripts/split_data.py
```

### 2. Fine-Tuning (v2)
Fine-tune the model on the T4 GPU.
```bash
python scripts/train.py
```
*Outputs save to `outputs/cibil_qwen2.5_lora_v2/`.*

### 3. Evaluation
Run the full production evaluation on the test set.
```bash
python scripts/evaluate_v2.py
```

### 4. Deployment & Export
Merge the LoRA weights and push to Hugging Face.
```bash
python scripts/export_model.py
```

---

## 🖥️ Production API Management

The API is configured to load V2 adapters from the `v2` branch of the Hugging Face repository.

### 1. Starting the Server
```bash
# Starts in background and logs to api_v2_hf.log
nohup python app.py > api_v2_hf.log 2>&1 &
```

### 2. Switching Versions (v1 vs v2)
To switch between versions, modify `app.py`:
- **v2**: `revision="v2"` in `PeftModel.from_pretrained`
- **v1**: `revision="v1"` in `PeftModel.from_pretrained`

### 3. Monitoring & Stopping
```bash
# View Logs
tail -f api_v2_hf.log

# Check Process
ps -aux | grep app.py

# Stop Server
pkill -f "python app.py"
```

---

## 📂 Directory Structure
- `scripts/`: Implementation scripts (v2 specific scripts included).
- `data/processed/`: Balanced 12k training/test datasets.
- `models/`: Merged standalone 15GB models (local only).
- `reports/`: Evaluation summaries and CSV results.

## 📝 License
Proprietary / Internal Use Only.
