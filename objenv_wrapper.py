import gym
import numpy as np
import torch.nn.functional as F
import crafter
import procgen
import cv2
import json
import torch
import torch.nn.functional as F
from hunter_game import Env as Hunter
from gym.spaces.utils import flatten_space, unflatten, flatdim, flatten 
from collections import OrderedDict

from space_wrapper import SpaceWrapper

def batch_flatten(space, x, batch=True):
  if isinstance(space, gym.spaces.Box):
      x = np.asarray(x, dtype=space.dtype)
      if batch:
        return x.reshape(x.shape[0], -1)
      else:
        return x.reshape(-1)
  elif isinstance(space, gym.spaces.Dict):
      return np.concatenate(
          [batch_flatten(s, x[key], batch) for key, s in space.spaces.items()], axis=-1)
  else:
      assert False

def batch_unflatten(space, x):
  assert isinstance(space, gym.spaces.Dict)
  assert len(x.shape) == 2
  dims = [flatdim(s) for s in space.spaces.values()]
  list_flattened = np.split(x, np.cumsum(dims)[:-1], axis=-1)
  list_unflattened = [
      (key, flattened.reshape((-1, *s.shape)))
      for flattened, (key,
                      s) in zip(list_flattened, space.spaces.items())
  ]
  return OrderedDict(list_unflattened)

class ObjCatPreprocess:
  def __init__(self, env, cfg):
    if not cfg.use_space:
      self.space = None
    else:
      cfg_path, kmeans_path = cfg.cfg_path, cfg.kmeans_path
      self.env = env
      self.space = SpaceWrapper(cfg_path, kmeans_path=kmeans_path)
      self.p = cfg.get('p', 0)
  def random_mask(self, p, obj_cat):
    if p == 0:
      return obj_cat
    else:
      mask = 1 - torch.distributions.bernoulli.Bernoulli(probs=torch.ones(list(obj_cat.shape[:-1]) + [1]) * p).sample()
      mask = mask.to(obj_cat.device)
      obj_cat[...,:-1] = obj_cat[...,:-1] * mask
      obj_cat[...,-1] = obj_cat[...,-1] + 1e-4
      obj_cat = obj_cat / obj_cat.sum(dim=-1, keepdim=True)
      return obj_cat
  def __call__(self, **kwargs):
    if not self.space:
      return kwargs
    for k in ['obs', 'obs_next']:
      if k not in kwargs:
        continue
      obs = self.env.batch_unflatten(kwargs[k])['obs'] 
      obj_cat = self.space.forward_kmeans(torch.as_tensor(obs))
      obj_cat = self.random_mask(self.p, obj_cat)
      new_obs = self.env.batch_flatten(dict(obs=obs, obj_cat=obj_cat))
      kwargs[k] = new_obs
    return kwargs

class ObjCatPreprocessV2:
  def __init__(self, env, cfg):
    if not cfg.use_space:
      assert False
      self.space = None
    else:
      cfg_path, kmeans_path = cfg.cfg_path, cfg.kmeans_path
      self.env = env
      self.space = SpaceWrapper(cfg_path, kmeans_path=kmeans_path)
    self.output_shape = (8 * 8, self.space.cfg.arch.z_what_dim + 2 + 1)  
    self.p = cfg.get('p', 0)
  def __call__(self, **kwargs):
    if not self.space:
      return kwargs
    for k in ['obs', 'obs_next']:
      if k not in kwargs:
        continue
      new_obs = torch.zeros(kwargs[k].shape)
      obs = self.env.batch_unflatten(kwargs[k])['obs'] 
      all_infos = self.space.forward(torch.as_tensor(obs), all_infos=True)
      z_pres, z_what, z_shift = [all_infos[k] for k in ['z_pres', 'z_what_ori', 'z_shift']]
      if self.p > 0:
        mask = 1 - torch.distributions.bernoulli.Bernoulli(probs=torch.ones(z_pres.shape) * self.p).sample()
        mask = mask.to(z_pres.device)
        z_pres = z_pres * mask
      obj_info = torch.cat([z_shift, z_what, z_pres], dim=-1).flatten(1)
      new_obs[...,:obj_info.shape[1]] = obj_info
      kwargs[k] = new_obs
    return kwargs

class ObjEnvWrapper(gym.Env):
  def __init__(self, env_name, cfg):
    self.cfg = cfg
    super().__init__()
    if 'crafter' in env_name:
      self.env_fn = lambda: crafter.Recorder(crafter.Env(), cfg.logdir,
                                      save_stats=True, save_video=False, save_episode=False)
    elif 'hunter' in env_name:
      self.env_fn = lambda: Hunter(**cfg.env_kwargs)
    else:
      assert False
    self.env = self.env_fn() 
    self.obj_cat_num = cfg.obj_cat_num
    self.G = cfg.get('G', 8)
    self.padding_zeros = np.zeros((self.G, self.G, self.obj_cat_num))
    self.padding_zeros[...,-1] = 1
    self.obs_space_dict = gym.spaces.Dict(OrderedDict(obs=self.env.observation_space,
                                               obj_cat=gym.spaces.Box(low=-1, high=1, shape=self.padding_zeros.shape)
                                                           ))
    self.observation_space = flatten_space(self.obs_space_dict)
    self.action_space = self.env.action_space
    self.reset_new = cfg.get('reset_new', False)
  def get_obs(self, o, obj_cat=None):
    if obj_cat is None:
      obj_cat = self.padding_zeros.copy()
    return batch_flatten(self.obs_space_dict, dict(obs=o, obj_cat=self.padding_zeros.copy()), False)
  def batch_unflatten(self, x):
    return batch_unflatten(self.obs_space_dict, x) 
  def batch_flatten(self, x, batch=True):
    return batch_flatten(self.obs_space_dict, x, batch) 
  def reset(self):
    if self.reset_new:
      self.env = self.env_fn()
    return self.get_obs(self.env.reset())
  def step(self, a):
    o, r, d, i = self.env.step(a)
    return self.get_obs(o), r, d, i
