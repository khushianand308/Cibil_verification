# Production & Development Commands

### 🐳 Docker (Production)
Run the API with GPU support and auto-restart using the production image.

**Image**: `khushianand28/cibil-api:latest`

**Start API**:
```bash
docker run --gpus all -d -p 9090:9090 --restart unless-stopped --ipc=host --shm-size=1g -v ~/.cache/huggingface:/root/.cache/huggingface --name cibil-production khushianand28/cibil-api:latest
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
Run the API directly on the host host using `nohup`.

**Start Forever**:
```bash
nohup python3 app.py > api.log 2>&1 &
```

**Stop Script**:
```bash
pkill -f "python3 app.py"
```

---

### 🚀 Maintenance (Old Image Cleanup)
To delete old images and save disk space, run this command regularly:

**Purge Everything (Old containers/images)**:
```bash
docker system prune -af
```

---

### 🚀 Manual CI/CD Deployment
Since automatic triggers are disabled, follow these steps to deploy a new version:

1.  Go to your GitHub Repository.
2.  Click on the **Actions** tab.
3.  Select **"Build and Push Docker Image"** from the sidebar.
4.  Click the **"Run workflow"** button on the right.
5.  Select the `main` branch and click **Run workflow**.

*This will automatically build, push to Docker Hub, and restart the API on your server.*

---

### 🧪 API Batch Test
**Test Command**:
```bash
curl -X POST "http://localhost:9090/verify-batch" -H "Content-Type: application/json" --data-binary @batch_test_real.json
```
