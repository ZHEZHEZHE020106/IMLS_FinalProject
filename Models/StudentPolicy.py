# Student Policy

import torch
import torch.nn as nn

class StudentPolicy(nn.Module):
    def __init__(self, encoder, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.encoder = encoder  # 可选冻结
        # 简化的LAM，只用一层
        self.lam = nn.Linear(encoder.embedding_dim, latent_dim)

    def forward(self, img):
        s = self.encoder(img)
        z = self.lam(s)
        return z