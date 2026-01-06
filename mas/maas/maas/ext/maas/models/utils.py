from sentence_transformers import SentenceTransformer
import torch
    
def get_sentence_embedding(sentence):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)

class SentenceEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sentence):
        embeddings = self.model.encode(sentence)
        return torch.tensor(embeddings)
    
def sample_operators(probs: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    device = probs.device
    probs = probs.detach()
    
    num_ops = probs.size(0)
    if num_ops == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    selected = torch.tensor([], dtype=torch.long, device=device)
    cumulative = 0.0
    remaining = torch.arange(num_ops, device=device)
    
    while cumulative < threshold and remaining.numel() > 0:
        sampled = torch.multinomial(probs[remaining], num_samples=1)
        idx = remaining[sampled].squeeze()

        if not torch.any(selected == idx):
            selected = torch.cat([selected, idx.unsqueeze(0)])
            cumulative += probs[idx].item()
        
        mask = torch.ones_like(remaining, dtype=torch.bool)
        mask[sampled] = False
        remaining = remaining[mask]
    
    if selected.numel() == 0:
        selected = probs.argmax().unsqueeze(0)
    
    return selected

