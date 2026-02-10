# Use Ubuntu 22.04 based CUDA image (has GLIBC 2.35)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 1. SETTINGS
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 2. SYSTEM DEPENDENCIES: Install Python 3.10 and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && python3 -m pip install --upgrade pip setuptools wheel

# 3. CONSOLIDATED AI STACK: Precise versions from your working host environment
# Installing everything in one go ensures binary compatibility for 'nms'
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio

RUN pip install --no-cache-dir \
    "transformers==4.57.6" \
    "peft==0.18.1" \
    "trl==0.24.0" \
    "bitsandbytes==0.49.1" \
    "accelerate==1.12.0" \
    "xformers==0.0.34"

# 4. PROJECT REQUIREMENTS
COPY requirements.txt .
# Remove unsloth from requirements to prevent conflict with our manual install
RUN sed -i '/unsloth/d' requirements.txt && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"

# 5. APPLICATION CODE
COPY . .

# 6. RUNTIME CONFIG
EXPOSE 9090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
