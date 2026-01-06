import torch
import torch.nn.functional as F
import numpy as np
from torch.special import gammaln
from maas.ext.maas.models.utils import SentenceEncoder, sample_operators

sentence_encoder = SentenceEncoder()

class OperatorSelector(torch.nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, device=None, is_first_layer: bool = False):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_first_layer = is_first_layer
        if self.is_first_layer:
            self.operator_encoder = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.operator_encoder = torch.nn.Linear(input_dim * 2, hidden_dim)
        self.query_encoder = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, query_embed: torch.Tensor, operators_embed: torch.Tensor, prev_operators_embed: torch.Tensor = None):
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0)

        query_embed = self.query_encoder(query_embed)   
        query_embed = F.normalize(query_embed, p=2, dim=1)

        if prev_operators_embed is not None and self.is_first_layer is False:
            prev_operator = prev_operators_embed[0].unsqueeze(0)
            prev_expanded = prev_operator.expand(operators_embed.size(0), -1)
            concat_embed = torch.cat([operators_embed, prev_expanded], dim=1)
            all_operators_embed = self.operator_encoder(concat_embed)
        else:
            all_operators_embed = self.operator_encoder(operators_embed)

        all_operators_embed = F.normalize(all_operators_embed, p=2, dim=1)       

        scores = torch.matmul(query_embed, all_operators_embed.T)

        probs = F.softmax(scores, dim=1)
        log_probs = F.log_softmax(scores, dim=1)

        return log_probs, probs

class MultiLayerController(torch.nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, num_layers: int = 4, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.layers = torch.nn.ModuleList([
            OperatorSelector(input_dim, hidden_dim, device=self.device, is_first_layer=(i == 0)) 
            for i in range(num_layers)
        ])

    def forward(self, query, operators_embedding, selection_operator_names, log_path=None):
        query_embedding = sentence_encoder(query).to(self.device)
        operators_embedding = operators_embedding.to(self.device) 
        log_probs_layers = []
        selected_names_layers = []
        prev_operators = None

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                log_probs, probs = layer(query_embedding, operators_embedding)
            else:
                log_probs, probs = layer(query_embedding, operators_embedding, prev_operators)
                
            probs_1d = probs.squeeze(0)
            log_probs_1d = log_probs.squeeze(0)

            selected_indices = sample_operators(probs_1d, threshold=0.3)
            selected_indices_list = selected_indices.cpu().tolist()
            selected_names = [selection_operator_names[idx] for idx in selected_indices_list]
            penalty_applied = False

            if layer_idx == 0:
                if any(name.lower() == "earlystop" for name in selected_names):
                    penalty_applied = True
                    try:
                        generate_idx = selection_operator_names.index("Generate")
                    except ValueError:
                        generate_idx = 0
                    selected_indices = torch.tensor([generate_idx], device=self.device)
                    selected_names = ["Generate"]
                elif not any("generate" in name.lower() for name in selected_names):
                    try:
                        generate_idx = selection_operator_names.index("Generate")
                    except ValueError:
                        generate_idx = 0
                    selected_indices = torch.tensor([generate_idx], device=self.device)
                    selected_names = ["Generate"]
                elif "generate" not in selected_names[0].lower() and any("generate" in name.lower() for name in selected_names):
                    for idx, name in enumerate(selected_names):
                        if "generate" in name.lower():
                            selected_names = [selected_names[idx]] + selected_names[:idx] + selected_names[idx+1:]
                            try:
                                new_first_idx = selection_operator_names.index(selected_names[0])
                            except ValueError:
                                new_first_idx = 0
                            new_indices = [new_first_idx] + [selection_operator_names.index(n) for n in selected_names[1:]]
                            selected_indices = torch.tensor(new_indices, device=self.device)
                            break

            if selected_indices.numel() > 0:
                layer_log_prob = torch.sum(log_probs_1d[selected_indices])
            else:
                layer_log_prob = torch.tensor(0.0, device=self.device)

            if layer_idx == 0 and penalty_applied:
                layer_log_prob = layer_log_prob + torch.tensor(-1.5, device=self.device)

            log_probs_layers.append(layer_log_prob)
            selected_names_layers.append(selected_names)

            if selected_indices.numel() > 0:
                selected_indices = selected_indices.to(operators_embedding.device)
                prev_operators = operators_embedding[selected_indices]
            else:
                prev_operators = None
            
            if (layer_idx == 0 and penalty_applied) or any(name.lower() == "earlystop" for name in selected_names):
                break

        return log_probs_layers, selected_names_layers

