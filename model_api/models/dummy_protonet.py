class DummyA2ProtoNet:
    def __init__(self):
        self.labels = ["FAKE", "REAL"]

    def predict(self, text: str):
        # Temporary dummy logic
        if len(text) % 2 == 0:
            return "FAKE", 0.73
        return "REAL", 0.81
