import torch.nn as nn
import torch

class TabularBERT(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        # Lookup tables for token meaning and position meaning
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        # Self-attention layers to analyze cross-layer syntax rules
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        out = self.embedding(x) + self.pos_embedding(positions)
        out = self.transformer(out)
        return self.fc_out(out)