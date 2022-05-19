import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
glimpse_size = 12
z_what_dim = 50
enc_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
out = enc_cnn(torch.ones(1, 3, glimpse_size, glimpse_size))
first_shape = out.shape[1:]
net = nn.Linear(np.prod(first_shape), z_what_dim)
dec = nn.Sequential(
  nn.Flatten(start_dim=-3),
  nn.Linear(z_what_dim, np.prod(first_shape)), nn.CELU(),
  Rearrange('n (c h w) -> n c h w', c=first_shape[0], h=first_shape[1], w=first_shape[2]),  #(32, 2, 2)
  nn.ConvTranspose2d(first_shape[0], 32, 2, stride=2), nn.CELU(), nn.GroupNorm(4,32),
  nn.ConvTranspose2d(32, 16, 2, stride=2), nn.CELU(), nn.GroupNorm(4,16),
  nn.Conv2d(16, 16, 3, padding=1), nn.CELU(), nn.GroupNorm(4,16),
)
z = net(out.flatten(start_dim=1))
out = dec(z.view(z.size(0), -1, 1, 1))
print(out.shape)

