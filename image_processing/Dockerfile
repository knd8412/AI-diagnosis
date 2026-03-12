FROM python:3.11-slim

WORKDIR /app/image_processing


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /root/.torchxrayvision/models_data && \
    wget -q "https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt" \
    -O "/root/.torchxrayvision/models_data/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"

COPY . .

ENV PYTHONPATH=/app/image_processing

EXPOSE 8001

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]
