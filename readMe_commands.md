### 🐳 Docker (Minimal Local Production)
Run the API with GPU support, auto-restart, and host-mounted models on **Port 5000**.

**Image**: `khushianand28/cibil-api:latest` (Minimal version)

**Start API (with Host Models)**:
> [!NOTE]
> Replace `/path/to/your/models` with the actual folder on your machine where the models are stored.

```bash
docker run --gpus all -d \
  -p 5000:5000 \
  --restart unless-stopped \
  --ipc=host \
  --shm-size=1g \
  -e WORKERS=2 \
  -v /path/to/your/models:/app/models \
  -e MODEL_NAME="/app/models/qwen2.5-7b" \
  -e ADAPTERS_REPO="/app/models/cibil-adapters" \
  --name cibil-production khushianand28/cibil-api:latest
```

**Stop & Remove API**:
```bash
docker stop cibil-production && docker rm cibil-production
```

**View Live Logs**:
```bash
docker logs -f cibil-production
```

---

### 🐍 Python (Forever/Background)
Run the API directly on the host using `nohup` on **Port 5000**.

**Start Forever (2 Workers)**:
```bash
export WORKERS=2
nohup python3 app.py > api.log 2>&1 &
```

**Stop Script**:
```bash
pkill -f "python3 app.py"
```

---

### 🚀 Maintenance
**Purge Everything (Old containers/images)**:
```bash
docker system prune -af
```

---

### 🚀 Manual CI/CD Deployment
1. Go to your GitHub Repository -> **Actions**.
2. Select **"Build and Push Docker Image"**.
3. Click **"Run workflow"** -> Select `main` -> **Run workflow**.

---

### 🧪 API Batch Test (Port 5000)
**Test Command**:
```bash
curl -X POST "http://localhost:5000/verify-batch" -H "Content-Type: application/json" --data-binary @batch_test_real.json
```
