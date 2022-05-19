import torch
import torch.nn as nn
# import satnet
import numpy as np
import gym
import torch.nn.functional as F
from encoder import *
from utils import *
import stable_baselines3.common.logger as L
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.utils.net.discrete import Actor, Critic
import itertools
from tianshou.utils.net.common import ActorCritic
import ppo_utils


class AddSInfo(nn.Module):
  def __init__(self, h, w, c, cout=32, channel_first=False, use_mlp=True):
    super().__init__()
    identity = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=torch.float32)
    grid = F.affine_grid(identity, [1, 1, h, w])
    grid = grid.permute(0, 3, 1, 2).contiguous()
    # (1, 2, h, w)
    self.register_buffer('grid', grid)
    assert channel_first == False
    if not channel_first:
      # (1, h, w, 2)
      self.grid = grid.permute(0,2,3,1)
    self.use_mlp = use_mlp
    if self.use_mlp:
      self.mlp = nn.Linear(c+2, cout)
  def forward(self, x):
    x = torch.cat([x, self.grid.to(x.device).expand(x.shape[0], -1, -1, -1)], dim=-1)
    if self.use_mlp:
      x = self.mlp(x)
    return x

class LastDimReshape(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = shape
  def forward(self, o):
    return o.reshape(*o.shape[:-1], *self.shape)

class SMORLEncoder(nn.Module):
  """
    input_shape: (n, c+1) = (n, z_where + z_what + z_pres)
  """
  def __init__(self, cfg, input_shape):
    super().__init__()
    self.cfg = cfg
    self.in_shape = input_shape
    E = self.in_shape[-1] - 1
    self.pre_trans = nn.Sequential(LastDimReshape(self.in_shape), NormAndTrans(norm=False, inds=[1,0,2]))
    # self.enc = QueryMultiHeadAttention(cfg.L, E, head_num=1)
    self.enc = QueryMultiHeadAttention(cfg.L, E, head_num=1, to_q_net=[32], to_k_net=[32], to_v_net=[32])
    self.post_trans = Rearrange('l b c -> b (l c)')
    self.output_dim = self.enc.L * E
  """
    x: (b, n * (c+1))
  """
  def forward(self, x, ret_latent=False, **kwargs):
    x = x[..., :np.prod(self.in_shape)]
    x = self.pre_trans(x) # n, b, c+1
    mask = einops.repeat(x[...,-1], 'n b -> b h l n', h=1, l=self.enc.L)
    x = x[...,:-1]
    out = self.enc(x, mask)
    out = self.post_trans(out)
    if ret_latent:
      return out, out 
    return out

