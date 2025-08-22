FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download weights at build time (optional but speeds cold start)
ENV HF_HOME=/root/.cache/huggingface
RUN python3 - <<'PY'
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModelForCausalLM
TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
PY

COPY app.py prompts.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
