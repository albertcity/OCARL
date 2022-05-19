from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
import stable_baselines3.common.logger as L
import functools
import gym
import numpy as np
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from encoder import *
import einops

class RNEncoder(nn.Module): 
  def __init__(self, obs_space, act_space, cfg):
    super().__init__()
    self.cfg = cfg
    obs_space = gym.spaces.Box(low=-1, high=1000, shape=cfg.obs_shape)
    self.enc = ImpalaEncoder(obs_space, channels=cfg.filters, flatten=False)
    c, h, w = self.enc.final_shape
    self.pred_z_cat = create_mlp(cfg.filters[-1], cfg.obj_cat_num, [cfg.filters[-1]], return_seq=True)
    self.output_shape = (h, w, c + cfg.obj_cat_num) 
  def split_obs(self, o):
    shape = o.shape
    obs_shape = self.cfg.obs_shape
    mask_shape = (8, 8, self.cfg.obj_cat_num)
    obs = o[...,:np.prod(obs_shape)].reshape(*shape[:-1], *obs_shape)
    mask = o[...,np.prod(obs_shape):].reshape(*shape[:-1], *mask_shape)
    return obs, mask.detach()
  def forward(self, x, ret_latent=False):
    if isinstance(x, dict):
      x = x['obs']
    obs, obj_cat = self.split_obs(x)

    out0 = self.enc(obs).permute(0,2,3,1) # (h, w, c)
    out = torch.cat([out0, obj_cat], dim=-1)
    if ret_latent:
      return out, out0
    else:
      return out
  def enc_loss(self, b, latent=None):
    if self.cfg.enc_coeff <= 0:
      pred_loss = torch.Tensor([0]).to(b.obs.device).sum()
    else:
      obs, obj_cat = self.split_obs(b.obs)
      if latent is None:
        latent = self.enc(obs)
      pred_z_cat = self.pred_z_cat(latent)
      pred_z_cat_loss = -(F.log_softmax(pred_z_cat, dim=-1) * obj_cat).sum(-1)
      pred_z_cat_loss = (pred_z_cat_loss).sum([1,2]).mean()
      L.record_mean('encoder/pred_loss', pred_z_cat_loss.item())
      pred_loss = self.cfg.enc_coeff * pred_z_cat_loss
    return pred_loss

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

class ObjSummary(nn.Module):
  def __init__(self, c, obj_cat_num):
    super().__init__()
    self.head = 4
    self.query_atten = QueryMultiHeadAttention(obj_cat_num, c, self.head,
                        to_q_net=[32], to_k_net=[32], to_v_net=[32], to_out_net=[])
    self.out_dim = c * obj_cat_num
  """
    x: (N, B, E)
    obj_cat: (N, B, S)
    out: (B, S*E)
  """
  def forward(self, x, obj_cat):
    mask = einops.repeat(obj_cat, 'n b s -> b h s n', h=self.head)
    out = self.query_atten(x, mask=mask)
    out = einops.rearrange(out, 's n e -> n (s e)')
    return out

class RNModule(nn.Module):
  def __init__(self, input_shape, action_space, cfg):
    super().__init__()
    self.cfg = cfg
    h, w, c = input_shape
    obj_cat_num = c - 32
    self.obj_cat_num = c - 32
    self.add_sinfo = AddSInfo(h, w, c, cout=32)
    self.trans = Rearrange('n h w c -> (h w) n c')

    self.atten = nn.MultiheadAttention(32, 4)
    if not cfg.use_sep_mlp:
      create_layer = nn.Linear
    else:
      create_layer = functools.partial(MultiLinear, num_linears=self.obj_cat_num)
    fdim = 32
    self.mlp = create_mlp(64, fdim, [64], create_layer=create_layer, return_seq=True)
    self.ac  = nn.Linear(fdim, action_space.n + 1)
  def forward(self, x, ret_atten_wts=False, mask_out = None):
    obj_cat = x[...,-self.obj_cat_num:] # B, H, W, S
    atten_wts = None
    x = self.add_sinfo(x)
    x = self.trans(x)
    atten_out, atten_wts = self.atten(x, x, x)
    x0 = x
    x = torch.cat([x, atten_out], dim=-1) # (N, B, 64)
    if self.cfg.use_sep_mlp:
      x = x.unsqueeze(-2).expand(-1, -1, self.obj_cat_num, -1) # (N, B, S, 64)
    out = self.mlp(x) 
    if self.cfg.use_sep_mlp:
      obj_cat = einops.repeat(obj_cat, 'b h w s -> (h w) b s k', k=1) # n, b, s, k
      if mask_out is not None:
        obj_cat = obj_cat * einops.repeat(to_torch_as(mask_out, obj_cat), 's -> s k', k=1)
        if True:
          obj_cat[...,-1,:] += 1e-4
          obj_cat = obj_cat / obj_cat.sum(-2, keepdim=True)
      out = (out * obj_cat).sum(-2) # N, B, 64 
    out = out.amax(0) # (n, 64)
    out = self.ac(out)
    if ret_atten_wts:
      return out, atten_wts
    return out
