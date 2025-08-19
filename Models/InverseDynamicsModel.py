# Inverse Dynamic Model

import torch
import torch.nn as nn

class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim=128, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, s1_embed, s2_embed):
        x = torch.cat([s1_embed, s2_embed], dim=1)
        return self.net(x)  # output latent action
