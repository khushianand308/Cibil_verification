# Production & Development Commands

### 🐳 Docker (Production)
Run the API with GPU support and auto-restart.

**Start Container**:
```bash
docker run --gpus all -d -p 9090:9090 --restart unless-stopped --ipc=host --shm-size=1g -v ~/.cache/huggingface:/root/.cache/huggingface --name cibil-production khushianand28/cibil-api:latest
```

**Stop & Remove**:
```bash
docker stop cibil-production && docker rm cibil-production
```

**View Logs**:
```bash
docker logs -f cibil-production
```

---

### 🐍 Python (Forever/Background)
Run the API directly on the host using `nohup`.

**Start Forever**:
```bash
nohup python3 app.py > api.log 2>&1 &
```

**Stop Script**:
```bash
pkill -f "python3 app.py"
```

---

### 🚀 CI/CD Maintenance (Manual)
The pipeline is located at `.github/workflows/docker-publish.yml`. 
*Manual trigger is enabled on GitHub Actions.*

**Purge Old Images**:
```bash
docker image prune -af
```
