from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class LACC(nn.Module):
    def __init__(self, vocab_size, max_len):
        super().__init__()
        self.encoder = torchvision.models.efficientnet_v2_m().features
        self.converter = nn.parameter.Parameter(torch.ones(64, vocab_size))

        self.silu = nn.SiLU()
        self.linear1 = nn.Linear(1280, 512)
        self.linear2 = nn.Linear(512, 64)
        self.linear3 = nn.Linear(64, max_len)

    def forward(self, x):
        feature = self.encoder(x)
        feature = torch.flatten(feature, start_dim=2)
        feature = torch.matmul(feature, self.converter)

        y = feature.transpose(-1, -2)
        y = self.linear1(y)
        y = self.silu(y)
        y = self.linear2(y)
        y = self.silu(y)
        y = self.linear3(y)

        return y
