### 🐳 Standard API (Port 5000)
Run the original reliable production API (30 RPM).

**Image**: `khushianand28/cibil-api:latest`

**Start Standard API**:
```bash
docker run --gpus all -d \
  -p 5000:5000 \
  --restart unless-stopped \
  --ipc=host \
  --shm-size=1g \
  -v ~/Cibil_verification/models:/app/models \
  --name cibil-production khushianand28/cibil-api:latest
```

---

### 🧪 API Testing (Port 5000)

**Single Verification**:
```bash
curl -s -X POST "http://localhost:5000/verify" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: Hi, am I speaking to PAWAN KUMAR? User: Yes, this is Pawan."}' | jq .
```

**Batch Verification**:
```bash
curl -s -X POST "http://localhost:5000/verify-batch" \
     -H "Content-Type: application/json" \
     -d '{
       "transcripts": [
         {
           "interaction_transcript": [
             {"role": "agent", "en_text": "Am I speaking to Pawan Kumar?"},
             {"role": "user", "en_text": "Yes, I am Pawan."}
           ]
         },
         {
           "interaction_transcript": [
             {"role": "agent", "en_text": "Is this Pawan?"},
             {"role": "user", "en_text": "No, wrong number."}
           ]
         }
       ]
     }' | jq .
```

---

### 🚀 Maintenance
**View Logs**: `docker logs -f cibil-production`
**Stop API**: `docker stop cibil-production && docker rm cibil-production`
**Stress Test**: `$HOME/miniconda3/bin/python stress_test.py`
