# Base Image with PyTorch & CUDA 12.1
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (git is needed for pip install git+...)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We install unsloth, fastapi, uvicorn and other essentials
RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "xformers==0.0.23.post1" \
    "trl<0.9.0" \
    "peft" \
    "accelerate" \
    "bitsandbytes" \
    "fastapi" \
    "uvicorn" \
    "pydantic"

# Copy application code
COPY app.py .
COPY scripts/inference.py . 
# (We copy inference.py just in case, though app.py has its own logic now)

# Expose the API port
EXPOSE 9090

# Start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
