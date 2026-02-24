# Cibil Verification API (Production)

This repository contains the production-ready Cibil Verification API, powered by a fine-tuned Qwen2.5-7B model (Unsloth/4-bit) specifically optimized for Indian financial context.

---

## 🚀 Quick Start (Production)

### 1. Start the API
Runs the container in the background with GPU support and auto-restart.
```bash
docker run --gpus all -d \
  -p 9090:9090 \
  --restart unless-stopped \
  --ipc=host \
  --shm-size=1g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name cibil-production \
  khushianand28/cibil-api:latest
```

### 2. Stop/Remove the API
```bash
docker stop cibil-production
docker rm cibil-production
```

---

## 🔄 CI/CD/CD Pipeline (Automation)

The project uses a fully automated pipeline via **GitHub Actions**.

### How it works:
1.  **Push to `main`**: Any code change pushed to the `main` branch triggers the pipeline.
2.  **Build**: GitHub builds a new Docker image (clearing space on the runner automatically).
3.  **Push**: The image is pushed to **Docker Hub** (`khushianand28/cibil-api:latest`).
4.  **Deploy**: GitHub logs into the production server via **SSH** and runs the update script.
5.  **Purge**: Old/unused Docker images are automatically deleted from the server to save disk space.

### Manual Update (If needed):
If you want to pull the latest image from the cloud manually:
```bash
docker pull khushianand28/cibil-api:latest
# Then restart following the "Start the API" command above.
```

---

## 🛠️ Management & Monitoring

### Check Logs (Real-time)
```bash
docker logs -f cibil-production
```

### Check GPU & Health
```bash
nvidia-smi
docker stats cibil-production
```

### Disk Cleanup (Manual Purge)
```bash
docker image prune -af
```

---

## 📡 API Usage

### Input Format (Structured JSON)
The API accepts the standard interaction transcript format.

**Endpoint**: `POST /verify`  
**Payload**:
```json
{
  "transcript": {
    "interaction_transcript": [
      {"role": "agent", "en_text": "Am I speaking to Shaik?"},
      {"role": "user", "en_text": "Yes, speaking."}
    ]
  }
}
```

### Output Format (camelCase)
```json
{
  "disposition": "ANSWERED",
  "loanNumberVerified": true,
  "nameVerified": true,
  "rpcStatus": "true"
}
```

---

## 🛡️ Required Secrets (GitHub)
To maintain the pipeline, these secrets must be set in GitHub Actions:
- `DOCKER_USERNAME`: `khushianand28`
- `DOCKER_PASSWORD`: Docker Hub Token
- `SSH_HOST`: Server IP
- `SSH_USERNAME`: `ubuntu`
- `SSH_PRIVATE_KEY`: ED25519 Private Key
