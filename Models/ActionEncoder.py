# ---- New Latent Action Decoder ----

import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    def __init__(self, latent_dim=32, stick_dim=5, num_buttons=20):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.stick_head = nn.Linear(64, stick_dim)           # stick action head
        self.button_head = nn.Linear(64, num_buttons)        # button action head

    def forward(self, z_q):
        x = self.shared(z_q)
        stick_pred = self.stick_head(x)           # shape: [B, 5]
        button_logits = self.button_head(x)       # shape: [B, 20]
        return stick_pred, button_logits
