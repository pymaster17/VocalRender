# Self-hosted VocalRender demo (single GPU). Not used by Hugging Face Spaces.
#
#   docker build -t vocalrender-demo .
#   docker run --gpus all -p 7860:7860 -v $HOME/.cache/huggingface:/root/.cache/huggingface vocalrender-demo
#
# Mount a local checkpoint directory and set VOCALRENDER_CKPT_DIR to skip the HF Hub download.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py"]
