import os

MODEL_NAME = os.getenv("MODEL_NAME", "a2_protonet")
ENABLE_DRIFT_MONITORING = os.getenv("ENABLE_DRIFT_MONITORING", "false")
