
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# ---- ResNet Encoder ----
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64x32x32
        self.layer1 = base.layer1                                     # 64x32x32
        self.layer2 = base.layer2                                     # 128x16x16
        self.layer3 = base.layer3                                     # 256x8x8
        self.layer4 = base.layer4                                     # 512x4x4

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]  # Skip connections

# ---- U-Net Decoder ----
class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=3):
        super().__init__()
        self.up3 = self._up_block(512 + latent_dim + 256, 256)
        self.up2 = self._up_block(256 + 128, 128)
        self.up1 = self._up_block(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, features, z_latent):
        B = z_latent.shape[0]
        z_feat = z_latent.view(B, -1, 1, 1).repeat(1, 1, 4, 4)  # match feature[3] (4x4)

        x = torch.cat([features[3], z_feat], dim=1)  # [B, 512+latent_dim, 4, 4]
        x = self.up3(torch.cat([x, self._resize(features[2], x)], dim=1))  # → 8×8
        x = self.up2(torch.cat([x, self._resize(features[1], x)], dim=1))  # → 16×16
        x = self.up1(torch.cat([x, self._resize(features[0], x)], dim=1))  # → 32×32
        x = F.interpolate(self.final(x), size=(64, 64), mode='bilinear', align_corners=False)
        return x

    def _resize(self, tensor, ref_tensor):
        return F.interpolate(tensor, size=ref_tensor.shape[-2:], mode='nearest')


# ---- Full FDM with ResNet-U-Net ----
class UNetFDM(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=False)
        self.decoder = UNetDecoder(latent_dim=latent_dim, out_channels=3)

    def forward(self, s1_image, z_latent):
        features = self.encoder(s1_image)
        out = self.decoder(features, z_latent)
        return out  # predicted s_{t+1} image
