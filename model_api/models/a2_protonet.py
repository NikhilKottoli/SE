import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class A2ProtoNet(nn.Module):
    def __init__(self, model_name="bert-base-uncased", labels=["FAKE", "REAL"]):
        super().__init__()
        self.labels = labels
        self.num_classes = len(labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.register_buffer(
            "prototypes",
            torch.zeros((self.num_classes, self.encoder.config.hidden_size))
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def encode(self, sentences):
        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)

        outputs = self.encoder(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0]

        return cls_embeddings
    
    def fgsm_attack(self, embeddings, epsilon=1e-3):
        embeddings = embeddings.clone().detach().requires_grad_(True)
        fake_target = torch.zeros(embeddings.size(0), dtype=torch.long).to(DEVICE)
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        loss = self.loss_fn(sim_matrix, fake_target)
        loss.backward()
        adv_embeddings = embeddings + epsilon * embeddings.grad.sign()
        return adv_embeddings.detach()
    
    def build_prototypes(self, support_sentences, support_labels):
        embeddings = self.encode(support_sentences)
        for c in range(self.num_classes):
            idx = [i for i, y in enumerate(support_labels) if y == c]
            if idx:
                self.prototypes[c] = embeddings[idx].mean(dim=0)
        return self.prototypes
    
    def training_step(self, support_texts, support_labels, optimizer, epsilon=1e-3):
        self.train()
        optimizer.zero_grad()
        clean_emb = self.encode(support_texts)
        adv_emb = self.fgsm_attack(clean_emb, epsilon)
        combined = torch.cat([clean_emb, adv_emb])         
        combined_labels = torch.tensor(support_labels * 2).to(DEVICE)
        self.build_prototypes(support_texts, support_labels)
        logits = self.classify(combined)
        loss = self.loss_fn(logits, combined_labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def classify(self, embeddings):
        distances = torch.cdist(embeddings, self.prototypes)
        return -distances 
    
    def predict(self, text: str):
        self.eval()
        with torch.no_grad():
            emb = self.encode([text])
            logits = self.classify(emb)
            probs = F.softmax(logits, dim=-1).cpu()

            idx = probs.argmax(dim=1).item()
            conf = float(probs[0][idx])

            return self.labels[idx], round(conf, 4)
