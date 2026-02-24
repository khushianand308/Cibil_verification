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

## 🖥️ Production API Management

The API runs on **Port 9090**. It is configured to load V2 adapters from the `v2` branch of the Hugging Face repository.

### 1. Starting the Server (Production Methods)

#### Which is better? TMUX vs nohup
- **TMUX (Recommended)**: Best for monitoring. You can "attach" to the session and see the same screen the server sees. Ideal if you want to watch the model load.
- **nohup**: Best for "fire and forget". It just runs in the background and writes to a log file. Use this if you don't need to interact with the console.

#### Option A: TMUX (Recommended for live monitoring)
TMUX ensures the API keeps running even if you disconnect from SSH, and allows you to "attach" to see logs live.
```bash
conda activate cibil_disposition

# Create a new session and start the API
tmux new -s cibil-v2 "./start_server.sh"

# To Detach (Leave it running in background): 
# Press Ctrl + B, then press D

# To Re-attach (See live logs again):
tmux attach -t cibil-v2
```

#### Option B: Normal Background (Standard nohup)
```bash
conda activate cibil_disposition

# Start in background using the startup script
nohup ./start_server.sh > api.log 2>&1 &
```

### 2. Monitoring & Troubleshooting
```bash
# Check if API is running and see Process ID (PID)
ps -aux | grep app.py

# Check GPU Memory Usage
nvidia-smi

# View Logs (If using Option B)
tail -f api_v2_hf.log
```

### 3. Killing / Stopping the API
```bash
# Stop any running app.py processes
pkill -f "python app.py"

# To kill a specific PID (if pkill fails)
# kill -9 <PID_FROM_PS_COMMAND>

# To kill the TMUX session
tmux kill-session -t cibil-v2
```

### 4. Switching Versions (v1 vs v2)
To switch between versions, modify `app.py`:
- **v2**: `model = PeftModel.from_pretrained(model, ADAPTERS_REPO, revision="v2")`
- **v1**: `model = PeftModel.from_pretrained(model, ADAPTERS_REPO, revision="v1")`

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

## � Docker Deployment (Production)

This project is fully containerized for easy deployment on any NVIDIA GPU server.

### 1. Build & Run
```bash
# Build the image locally
docker build -t cibil-api .

# Run the container with GPU access and speed optimizations
docker run --gpus all -d \
  -p 9090:9090 \
  --ipc=host \
  --shm-size=1g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.triton:/root/.triton \
  --name cibil-production \
  cibil-api
```

### 2. 🔄 Dynamic Versioning
You can change the model or adapter version without rebuilding the image by using environment variables:

```bash
# Example: Run with v3 adapters
docker run --gpus all -d -p 9090:9090 \
  -e ADAPTERS_REVISION="v3" \
  --name cibil-v3 cibil-api
```

- `ADAPTERS_REPO`: HuggingFace repository for adapters.
- `ADAPTERS_REVISION`: Branch/Tag (e.g., `v2`, `v3`).

### 3. 🚀 Sharing the Image

#### Option A: Docker Hub (Recommended)
```bash
# Tag the local image
docker tag cibil-api:latest khushianand28/cibil-api:v2.0

# Push to your repository
docker push khushianand28/cibil-api:v2.0

# To run on another system:
docker pull khushianand28/cibil-api:v2.0
docker run --gpus all -d -p 9090:9090 --name cibil-api khushianand28/cibil-api:v2.0
```

#### Option B: Tar Archive
```bash
# Save to file
docker save -o cibil-api.tar cibil-api

# Load on new machine
docker load -i cibil-api.tar
```

### 4. 🛬 Run on Another System
If you have pushed the image to Docker Hub, anyone can run it on a new system with these two steps:

**Step 1: Pull the image**
```bash
docker pull khushianand28/cibil-api:v2.0
```

**Step 2: Run it**
```bash
docker run --gpus all -d \
  -p 9090:9090 \
  --ipc=host \
  --shm-size=1g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.triton:/root/.triton \
  --name cibil-production \
  khushianand28/cibil-api:v2.0
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
