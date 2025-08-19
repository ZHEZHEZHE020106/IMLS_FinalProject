# IMPALA-CNN Encoder

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=128, channel_multiplier=1):
        super().__init__()
        c1, c2, c3 = 16 * channel_multiplier, 32 * channel_multiplier, 32 * channel_multiplier
        self.embedding_dim = embedding_dim
        self.conv_seq = nn.Sequential(
            ImpalaBlock(in_channels, c1),
            ImpalaBlock(c1, c2),
            ImpalaBlock(c2, c3),
            nn.ReLU(),
            nn.Flatten()
        )
        # Assuming input image is 64x64, after 3 strided pools → size reduces to 8x8
        final_feat_dim = c3 * 8 * 8
        self.fc = nn.Linear(final_feat_dim, embedding_dim)

    def forward(self, x):
        x = self.conv_seq(x)
        x = self.fc(x)
        return x
