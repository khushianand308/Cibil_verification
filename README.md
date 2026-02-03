# Cibil Verification Model Fine-Tuning

This project contains the pipeline and scripts for fine-tuning a Small Language Model (SLM) to extract structured verification details from call transcripts. 

## 🎯 Project Goal
The objective is to analyze customer call transcripts and output a structured JSON containing:
1. `DISPOSITION`: The final status of the call.
2. `NAME_VERIFIED`: Boolean flag if the customer name was confirmed.
3. `LOAN_NUMBER_VERIFIED`: Boolean flag if the loan details were confirmed.
4. `RPC_STATUS`: Right Party Contact status (true, false, partial, insufficient_data).

## 🚀 Performance Snapshot (Test Set)
The model was fine-tuned on **2,840 balanced samples** using `Qwen2.5-7B-Instruct`.

| Metric | Score | status |
| :--- | :--- | :--- |
| **Overall Accuracy (Exact Match)** | **84.15%** | ✅ Pass |
| **JSON Validity Rate** | **100.00%** | ✅ Pass |
| **Final Train Loss** | **0.38** | ✅ Stable |

### Field-Level Metrics:
- **Loan Number Verification**: 98.24%
- **Name Verification**: 96.13%
- **Disposition Accuracy**: 89.08%
- **RPC Status Accuracy**: 88.73%

---

## 🛠️ Environment Setup
The project uses the `disposition_v3` conda environment.

```bash
conda activate disposition_v3
```

**Key Dependencies**:
- `unsloth`: For 2x faster, 4-bit memory-efficient fine-tuning.
- `transformers`, `peft`, `trl`: Hugging Face training stack.
- `torch`: Deep learning framework.

---

## 🏃 Execution Flow

### 1. Data Processing
Converts raw CSV data into balanced, ChatML-formatted JSONL files.
```bash
python scripts/preprocess_data.py
python scripts/balance_data.py
python scripts/split_data.py
```

### 2. Fine-Tuning (Training)
Uses QLoRA to train the adapters.
```bash
python scripts/train.py
```
*Outputs are saved to `outputs/cibil_qwen25_lora/`.*

### 3. Inference & Testing
Run a quick test on sample transcripts:
```bash
python scripts/inference.py
```

### 4. Full Evaluation
Calculate metrics on the unseen test set:
```bash
python scripts/evaluate_model.py
```

---

## 📂 Directory Structure
- `scripts/`: Implementation scripts for end-to-end lifecycle.
- `data/processed/`: Tokenized and balanced datasets.
- `outputs/cibil_qwen25_lora/`: Final fine-tuned LoRA adapters.
- `training.log`: Detailed logs from the latest training run.

## 📝 License
Proprietary / Internal Use Only.
