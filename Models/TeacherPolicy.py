import torch
import torch.nn as nn

class TeacherPolicy(nn.Module):
    def __init__(self, encoder, vq, latent_dim=32, hidden_dim=128, action_dim=8):
        super().__init__()
        self.encoder = encoder    # 冻结或可选微调
        self.vq = vq              # 冻结VQ
        # 正向Latent Action Model (LAM)
        self.lam = nn.Sequential(
            nn.Linear(encoder.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Action Decoder: latent -> 连续摇杆 + 离散按键 logits
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.stick_head = nn.Linear(hidden_dim, action_dim - 4)  # 连续摇杆维度
        self.button_head = nn.Linear(hidden_dim, 4)              # 4个离散按钮

    def forward(self, img):
        # 1. 图像编码
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