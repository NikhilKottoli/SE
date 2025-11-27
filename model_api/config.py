import os

MODEL_NAME = os.getenv("MODEL_NAME", "dummy_protonet")
ENABLE_DRIFT_MONITORING = os.getenv("ENABLE_DRIFT_MONITORING", "false")
