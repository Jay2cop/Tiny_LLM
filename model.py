import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, block_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True), 
            num_layers=n_layers
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        x = tok + pos
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.blocks(x, mask=mask)
        x = self.ln(x)
        logits = self.head(x)
        return logits
