#!/bin/bash

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate disposition_v3

# Run the server
# --reload enables auto-restart when you change code (great for dev)
echo "🚀 Starting Cibil Verification API on Port 9090..."
uvicorn app:app --host 0.0.0.0 --port 9090
