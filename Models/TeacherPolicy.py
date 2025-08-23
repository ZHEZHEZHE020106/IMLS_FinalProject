import torch
import torch.nn as nn
"""
class TeacherPolicy(nn.Module):
    def __init__(self, encoder, vq, latent_dim=32, hidden_dim=128, action_dim=8):
        super().__init__()
        self.encoder = encoder    
        self.vq = vq              
        # Latent Action Model (LAM)
        self.lam = nn.Sequential(
            nn.Linear(encoder.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Action Decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.stick_head = nn.Linear(hidden_dim, action_dim - 4)  
        self.button_head = nn.Linear(hidden_dim, 4)              

    def forward(self, img):
        # 1. image encoder
        s = self.encoder(img)
        # 2. VQ to latent
        z_q, _ = self.vq(s)
        # 3. Latent Action
        z = self.lam(z_q)
        # 4. Decode action
        hid = self.action_decoder(z)
        stick = self.stick_head(hid)
        button_logits = self.button_head(hid)
        return z, stick, button_logits
"""  

class TeacherPolicy(nn.Module):
    def __init__(self, encoder, latent_dim=32, hidden_dim=128, action_dim=8):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.embedding_dim * 2
        # Latent Action Model (LAM)
        self.lam = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Action Decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.stick_head = nn.Linear(hidden_dim, action_dim - 4)  
        self.button_head = nn.Linear(hidden_dim, 4)              

    def forward(self, img1, img2):
        # 1. image encoder
        s1, s2 = self.encoder(img1), self.encoder(img2)

        x = torch.cat([s1, s2], dim=-1)
        # 3. Latent Action
        z = self.lam(x)
        # 4. Decode action
        hid = self.action_decoder(z)
        stick = self.stick_head(hid)
        button_logits = self.button_head(hid)
        return z, stick, button_logits