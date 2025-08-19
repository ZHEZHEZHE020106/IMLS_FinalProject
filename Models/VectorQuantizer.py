# VectorQuantizerEMA

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.embedding)
            + torch.sum(self.embedding ** 2, dim=0, keepdim=True)
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.t()).view(z.shape)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_((1 - self.decay) * encodings.sum(0))
            embed_sum = torch.matmul(z_flattened.t(), encodings)
            self.embed_avg.data.mul_(self.decay).add_((1 - self.decay) * embed_sum)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            self.embedding.data = self.embed_avg / cluster_size.unsqueeze(0)

        quantized = z + (quantized - z).detach()
        loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        return quantized, loss
