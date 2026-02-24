# Cibil Verification API: Production Commands

### 🐳 Docker Commands (Recommended)
This is the easiest way to run the API with GPUs and auto-restart.

**Start API**:
```bash
docker run --gpus all -d -p 9090:9090 --restart unless-stopped --name cibil-production khushianand28/cibil-api:latest
```

**Stop API**:
```bash
docker stop cibil-production && docker rm cibil-production
```

**Check Logs**:
```bash
docker logs -f cibil-production
```

---

### 🐍 Python Commands (Local)
If you want to run the script directly on the server host and make it run "forever":

**Start API (Background/Forever)**:
```bash
nohup python3 app.py > api.log 2>&1 &
```

**Stop API**:
```bash
pkill -f "python3 app.py"
```

---

### 🚀 CI/CD Automation
The pipeline script is located at: `.github/workflows/docker-publish.yml`

*Pushing to the **main** branch automatically builds, pushes to Docker Hub, and restarts the server.*

---

### 📡 Test Commands

**Batch Test (Multiple Transcripts)**:
```bash
curl -X POST "http://localhost:9090/verify-batch" -H "Content-Type: application/json" -d '{"transcripts": [{"interaction_transcript": [{"role": "agent", "en_text": "Hello?"}]}]}'
```
