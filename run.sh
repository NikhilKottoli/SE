#!/bin/bash

# Create root project directory
mkdir -p your-project
cd your-project

##############################################
# docker-compose.yaml
##############################################
cat << 'EOF' > docker-compose.yaml
version: "3.9"

services:
  model-api:
    build:
      context: ./model_api
      dockerfile: Dockerfile
    container_name: model_api
    ports:
      - "8000:8000"
    environment:
      MODEL_NAME: a2_protonet
      ENABLE_DRIFT_MONITORING: "true"
      LOG_LEVEL: info
    volumes:
      - model_cache:/root/.cache/
    restart: unless-stopped

  worker:
    build:
      context: ./worker
      dockerfile: Dockerfile
    container_name: adaptation_worker
    environment:
      ATTACK_TOOL: textfooler
      DRIFT_THRESHOLD: "0.65"
    volumes:
      - worker_cache:/root/.cache/
    restart: on-failure

  monitoring:
    image: grafana/grafana:10.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "admin"
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana_data:
  model_cache:
  worker_cache:
EOF


##############################################
# model_api files
##############################################
mkdir -p model_api/models
mkdir -p model_api/utils
mkdir -p model_api/tests

# main.py
cat << 'EOF' > model_api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import time
from models.dummy_protonet import DummyA2ProtoNet

app = FastAPI(title="A2-ProtoNet Fake News API")

model = DummyA2ProtoNet()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    processing_time_ms: float

@app.get("/")
def root():
    return {"status": "API running", "model": "dummy A2-ProtoNet"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.time()
    label, conf = model.predict(req.text)
    elapsed = (time.time() - start) * 1000
    return PredictionResponse(
        label=label,
        confidence=conf,
        processing_time_ms=round(elapsed, 3)
    )
EOF

# dummy model
cat << 'EOF' > model_api/models/dummy_protonet.py
class DummyA2ProtoNet:
    def __init__(self):
        self.labels = ["FAKE", "REAL"]

    def predict(self, text: str):
        # Temporary dummy logic
        if len(text) % 2 == 0:
            return "FAKE", 0.73
        return "REAL", 0.81
EOF

# utils
cat << 'EOF' > model_api/utils/preprocess.py
def clean_text(text: str) -> str:
    return text.lower().strip()
EOF

# config
cat << 'EOF' > model_api/config.py
import os

MODEL_NAME = os.getenv("MODEL_NAME", "dummy_protonet")
ENABLE_DRIFT_MONITORING = os.getenv("ENABLE_DRIFT_MONITORING", "false")
EOF

# tests
cat << 'EOF' > model_api/tests/test_api.py
def test_dummy():
    assert 1 == 1
EOF

# Dockerfile
cat << 'EOF' > model_api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# requirements.txt
cat << 'EOF' > model_api/requirements.txt
fastapi
uvicorn
pydantic
EOF


##############################################
# worker files
##############################################
mkdir -p worker/drift
mkdir -p worker/attacks

# worker.py
cat << 'EOF' > worker/worker.py
import time

def run_worker():
    while True:
        print("Worker: Running scheduled tasks...")
        time.sleep(10)

if __name__ == "__main__":
    run_worker()
EOF

# drift detector
cat << 'EOF' > worker/drift/drift_detector.py
def detect_drift():
    print("Drift detection placeholder - replace later")
EOF

# textfooler
cat << 'EOF' > worker/attacks/textfooler.py
def attack(text: str):
    return text + " [perturbed]"
EOF

# worker Dockerfile
cat << 'EOF' > worker/Dockerfile
FROM python:3.11-slim

WORKDIR /worker

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "worker.py"]
EOF

# worker requirements
cat << 'EOF' > worker/requirements.txt
EOF


##############################################
# monitoring scaffolding
##############################################
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/provisioning/datasources

cat << 'EOF' > monitoring/README.md
Grafana setup placeholder.
EOF


echo "Project scaffolding created successfully!"

