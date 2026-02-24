# Use minimal Python 3.10 slim image
FROM python:3.10-slim

# 1. SETTINGS
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 2. SYSTEM DEPENDENCIES: Minimal tools for pip installations
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. AI STACK: Runtime libraries include CUDA support in wheels
# Consolidating to minimize layers
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==2.10.0 \
    torchvision==0.25.0 \
    torchaudio \
    "transformers==4.57.6" \
    "peft==0.18.1" \
    "trl==0.24.0" \
    "bitsandbytes==0.49.1" \
    "accelerate==1.12.0" \
    "xformers==0.0.34"

# 4. PROJECT REQUIREMENTS
COPY requirements.txt .
# Remove items already installed in the stack above to save time
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/transformers/d' requirements.txt && \
    sed -i '/peft/d' requirements.txt && \
    sed -i '/accelerate/d' requirements.txt && \
    sed -i '/bitsandbytes/d' requirements.txt && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"

# 5. APPLICATION CODE
COPY . .

# 6. RUNTIME CONFIG
EXPOSE 5000

# Default to 1 worker for safety. 2 workers recommended for 16GB VRAM.
ENV WORKERS=1

CMD ["python3", "app.py"]
