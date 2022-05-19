import numpy as np
import torch 
import gym
import procgen
import torch.nn as nn
import os
import torch.nn.functional as F
import einops
import cv2
import joblib
from space.model import get_model
# from space.config import cfg
from space.utils import Checkpointer
import torchvision.transforms.functional as VF
import omegaconf as oc

class SpaceWrapper(object):
  def __init__(self, cfg_path, ckpt_path=None, kmeans_path=None, obj_thres=0.8):
    super().__init__()
    cfg = oc.OmegaConf.load(cfg_path)
    ckpt_path = cfg.checkpointdir if not ckpt_path else ckpt_path
    self.cfg = cfg
    self.model_input_size = cfg.arch.img_shape
    model = get_model(cfg)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = Checkpointer(ckpt_path, 1).load_last('', model, None, None, use_cpu=(self.device == 'cpu'))
    self.iter_n = ckpt.pop('global_step')
    self.model= model
    self.fg = model.fg_module.to(self.device)
    self.fg.eval()
    if kmeans_path:
      print('KMEANS Loaded.')
      self.clf = joblib.load(kmeans_path) 
      self.num_cat = len(self.clf['kmeans'].cluster_centers_)
    else:
      self.num_cat = self.arch.num_cat

    self.obj_thres = obj_thres

  """
    z_shift, z_pres, z_what: (B, G, G, n), n=2,1,num_cat
  """
  @torch.no_grad()
  def postprocessV2(self, z_shift, z_pres, z_what):
      G = 8
      B = z_pres.shape[0]
      z_shift = ((z_shift + 1) * G // 2).long().clamp(0, G-1)
      z_shift = z_shift[...,1] * G + z_shift[...,0]
      z_shift_one_hot = F.one_hot(z_shift.reshape(B, G**2), G**2) # (B, G**2, G**2)
      z_pres = z_pres.reshape(B, G**2, -1).float()
      z_what = z_what.reshape(B, G**2, -1).float()
      valid_mask = z_pres >= self.obj_thres
      z_shift_one_hot = z_shift_one_hot * valid_mask # (B, G**2, G**2)
      z_shift_one_hot = z_shift_one_hot.permute(0,2,1).float()
      new_z_pres = torch.matmul(z_shift_one_hot, z_pres)
      new_z_what = torch.matmul(z_shift_one_hot, z_what)
      new_z_pres = new_z_pres.reshape(B, G,G, -1)
      new_z_what = new_z_what.reshape(B, G,G, -1)
      return new_z_pres, new_z_what

  @torch.no_grad()
  def forward_kmeans(self, x):
    x = torch.as_tensor(x, device = self.device).float().permute(0,3,1,2) / 255.
    B = x.shape[0]
    G = 8
    res = self.fg(x, 1000000, glimpse_only=True)
    z_pres, glimpse, z_shift = res['z_pres'], res['glimpse'].detach().cpu(), res['z_shift'] 

    z_cat = torch.zeros(B*G*G, self.num_cat)
    valid_mask = (z_pres >= self.obj_thres).detach().cpu()
    valid_glimpse = glimpse[valid_mask]
    if len(valid_glimpse) > 0:
      valid_glimpse = valid_glimpse.flatten(1)
      valid_cat = self.clf['kmeans'].predict(self.clf['pca'].transform(valid_glimpse))
      z_cat[valid_mask] = F.one_hot(torch.as_tensor(valid_cat).long(), self.num_cat).float()
    z_cat = z_cat.reshape(B, G, G, -1).to(self.device)

    z_shift = z_shift.reshape(B, G, G, -1)
    z_pres = z_pres.reshape(B, G, G, -1)
    valid_mask, z_cat = self.postprocessV2(z_shift, z_pres, z_cat) 
    valid_mask = valid_mask.clamp(0, 1)
    z_cat = torch.cat([z_cat, 1 - valid_mask], dim=-1)
    z_cat = z_cat / z_cat.sum(-1, keepdim=True)
    return z_cat.detach().cpu()

  """
    x: (B, H, W, C) 
    out: (B, H//8, W//8, obj_type + 1)
  """
  @torch.no_grad()
  def forward(self, x, all_infos=False, use_smooth=False):
    # x = np.asarray(x)
    x = torch.as_tensor(x, device = self.device).float().permute(0,3,1,2) / 255.
    B = x.shape[0]
    G = 8
    if self.model_input_size[0] != 64:
      x = VF.resize(x, self.model_input_size) 
    fg = self.fg(x, 10000000, encode_only=True) 
    z_shift, z_what0, z_pres, z_what_cat = [fg[k].reshape(B, G, G, -1) for k in ['z_shift', 'z_what', 'z_pres', 'z_what_cat']] # (B, G, G, n)
    valid_mask = (z_pres >= self.obj_thres).float()
    if not use_smooth:
      valid_mask, z_what = self.postprocessV2(z_shift, valid_mask, z_what_cat)
      valid_mask = valid_mask.clamp(0, 1)
      z_what = torch.cat([z_what, 1-valid_mask], dim=-1)
    else:
      z_what = self.SmoothPostprocess(z_shift, z_pres, z_what_cat)
      #z_what = self.smooth(z_what)
      z_what[...,-1] = 0.01
    z_what = z_what / z_what.sum(-1, keepdim=True)
    if all_infos:
      return dict(z_pres=z_pres.detach().cpu(), valid_mask=valid_mask.detach().cpu(),
                  z_what=z_what.detach().cpu(), z_what_ori=z_what0.detach().cpu(), ori_z_what = z_what_cat.detach().cpu(), z_shift=z_shift.detach().cpu())
    return z_what
