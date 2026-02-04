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

---

## 🖥️ Production Server Management

### 1. Starting the Server
The server runs on **Port 9090**.

**Option A: Using the automation script (Recommended)**
```bash
./start_server.sh
```

**Option B: Manual Start (via Conda)**
```bash
# Activate the environment
conda activate disposition_v3

# Method 1: Direct Python (Recommended for testing)
python app.py

# Method 2: Uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 9090
```

### 2. Stopping / Killing the Server
```bash
# Graceful stop (if running in terminal)
Ctrl + C

# Forced stop (for background processes)
pkill -f "uvicorn app:app"
```

### 3. Running in Background (24/7)
To keep the server alive after you close your laptop, use **tmux**.
```bash
# Start a new session and run server
tmux new -s cibil-server "./start_server.sh"

# Disconnect (keep running)
Press Ctrl + B, then D

# Re-attach to check logs
tmux attach -t cibil-server

# Kill / Close the session entirely
tmux kill-session -t cibil-server
```

---

## 🧪 API Testing (CURL Commands)

The main endpoint is `POST /verify`.

### Scenario 1: Full Verification
```bash
curl -X POST "http://localhost:9090/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: HI, I am Sakshi from HDB. Am I speaking to VIJAYAKRISHNA?\nUser: Yeah.\nAgent: Confirm loan 21?\nUser: Yes."}'
```

### Scenario 2: Wrong Number
```bash
curl -X POST "http://localhost:9090/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: Am I speaking to SRUTHI?\nUser: No, this is a wrong number. I am not Sruthi."}'
```

### Scenario 3: Disconnected
```bash
curl -X POST "http://localhost:9090/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: Hello? Is anyone there?\nAgent: I think the line is bad, I will call you back."}'
```

## 📝 License
Proprietary / Internal Use Only.
