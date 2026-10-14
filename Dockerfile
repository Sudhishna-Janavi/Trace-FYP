# Trace. One image that runs on Fly, Hugging Face Spaces, Render or anywhere
# else that takes a container.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    ONNX_THREADS=1

WORKDIR /app

# Dependencies first, so editing the application does not reinstall torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The ONNX graph is baked in: the application is offline at runtime, and a
# missing model should fail the build rather than the first user request.
COPY scripts/fetch_model.py scripts/fetch_model.py
RUN python scripts/fetch_model.py

COPY backend/ backend/
COPY models/ models/
COPY dataset/demo/ dataset/demo/
COPY scripts/ scripts/

EXPOSE 7860
# One worker. Each would load its own copy of the encoder, and the memory is
# the binding constraint on every small instance.
CMD ["sh", "-c", "python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
