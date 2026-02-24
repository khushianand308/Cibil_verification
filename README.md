# Cibil Verification API (Production v2.1)

Minimal inference API for extracting structured verification details from call transcripts.

## 🎯 Features
- **GPU Accelerated**: Optimized with Unsloth for fast inference.
- **Concurrent Processing**: Multi-worker support via Uvicorn.
- **Batch Processing**: `/verify-batch` endpoint for high-throughput.
- **Minimal Footprint**: Slim Docker image using host-mounted models.

---

### 🚀 Production Status
- **Accuracy**: 89.94% (Exact Match)
- **Rules Compliance**: 99.03%
- **Port**: 5000 (Default)

---

### 🏃 Quick Start (API)

**1. Start the Container** (Port 5000):
```bash
docker run --gpus all -d -p 5000:5000 \
  -e WORKERS=2 \
  -v /path/to/models:/app/models \
  --name cibil-api khushianand28/cibil-api:latest
```

**2. Test with CURL**:
```bash
curl -X POST "http://localhost:5000/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: HI. User: Yes."}'
```

---

### 📄 Documentation
- Detailed admin commands (start/stop/clean): [readMe_commands.md](file:///home/ubuntu/Cibil_verification/readMe_commands.md)
- Deployment workflow: [GitHub Actions Workflow](file:///home/ubuntu/Cibil_verification/.github/workflows/docker-publish.yml)

## 📂 Minimal Structure
- `app.py`: Core FastAPI application.
- `Dockerfile`: Optimized slim production image.
- `requirements.txt`: Minimal runtime dependencies.
- `.dockerignore`: Excludes all training data/scripts for minimal builds.
