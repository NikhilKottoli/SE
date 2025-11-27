from fastapi import FastAPI, Response
from pydantic import BaseModel
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from models.dummy_protonet import DummyA2ProtoNet

app = FastAPI(title="A2-ProtoNet Fake News API")

model = DummyA2ProtoNet()

# Manual Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    processing_time_ms: float

@app.get("/")
def root():
    return {"status": "API running", "model": "dummy A2-ProtoNet"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.time()
    label, conf = model.predict(req.text)
    elapsed = (time.time() - start) * 1000
    
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    REQUEST_DURATION.observe(elapsed / 1000)
    
    return PredictionResponse(
        label=label,
        confidence=conf,
        processing_time_ms=round(elapsed, 3)
    )