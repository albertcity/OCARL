import torch
import torch.nn as nn
# import satnet
import numpy as np
import gym
import torch.nn.functional as F
from encoder import *
from utils import *
# from stable_baselines3.common.torch_layers import create_mlp
import stable_baselines3.common.logger as L
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.utils.net.discrete import Actor, Critic
import itertools
from tianshou.utils.net.common import ActorCritic
import ppo_utils
from tianshou.policy import BasePolicy, PPOPolicy

class PPOBase(PPOPolicy):
  def __init__(self, obs_space, act_space,
              latent_dim = 256, net_arch=[], device='cpu',
              actor_fn=None, critic_fn=None,
              make_enc_ac=None, **kwargs):
    self.device = device
    self.kwargs = kwargs
    self.obs_space = obs_space
    self.act_space = act_space
    self.latent_dim = latent_dim
    nn.Module.__init__(self)
    if make_enc_ac is None:
      self.enc = ImpalaEncoder(obs_space, latent_dim, activation_fn=nn.ReLU, lnorm=False)
      self.final_shape = self.enc.final_shape
      self.actor_critic = nn.Sequential(*create_mlp(latent_dim, act_space.n+1, net_arch))
    else:
      self.enc, self.actor_critic = make_enc_ac(obs_space, act_space)
    optim = torch.optim.Adam(itertools.chain(self.enc.parameters(), self.actor_critic.parameters()), lr=5e-4)
    dist = torch.distributions.Categorical
    PPOPolicy.__init__(self, self.actor_fn if actor_fn is None else actor_fn, self.critic_fn if critic_fn is None else critic_fn, optim, dist, init_module=False, **kwargs)
    self.to(device)

  def actor_fn(self, obs, state, **kwargs):
    res = self.forward(Batch(obs=obs))
    return res.probs, state
  def critic_fn(self, obs):
    return self.forward(Batch(obs=obs)).fvalues
  def forward(self, batch, state=None, **kwargs):
    latent = self.forward_latent(batch, **kwargs)
    return self.forward_pol(latent, self.actor_critic)
  def forward_latent(self, batch, ret_latent=False):
    obs = to_torch(batch.obs, torch.float32, self.device)
    latent = self.enc(obs, ret_latent=ret_latent)
    return latent
  def forward_pol(self, latent, pol, state=None):
    latent = pol(latent)
    logits, fvalues = latent[...,1:], latent[...,:1]
    probs = F.softmax(logits, -1)
    dist = self.dist_fn(probs)
    if self._deterministic_eval and not self.training:
        if self.action_type == "discrete":
            act = probs.argmax(-1)
        elif self.action_type == "continuous":
            act = probs[0]
    else:
        act = dist.sample()
    return Batch(probs=probs, act=act, state=state, dist=dist, fvalues=fvalues,
                policy=Batch(probs=probs, fvalues=fvalues))
  def log(self, loss_info, loss_infos, suffix=''):
    for k, v in loss_info.items():
      ks = k + suffix
      if ks not in loss_infos:
        loss_infos[ks] = []
      loss_infos[ks].append(v)
      L.record_mean(ks, v)
    return loss_infos
  def learn(self, batch, batch_size, repeat, **kwargs):
      loss_infos = {}
      for step in range(repeat):
          if self._recompute_adv and step > 0:
              batch = self._compute_returns(batch, self._buffer, self._indices)
          for b in batch.split(batch_size, merge_last=True):
              b = to_torch(b, torch.float32, self.device)
              res, latent = self.forward_latent(b, ret_latent=True)
              res = self.forward_pol(res, self.actor_critic)
              train_loss, train_loss_info = ppo_utils.ppo_loss(self.dist_fn(res.probs), res.fvalues,
                                          *[b.__dict__[attr] for attr in ['adv', 'act', 'logp_old', 'v_s', 'returns']],
                                          self)
              if hasattr(self.enc, 'enc_loss'):
                # b = to_torch(b, torch.float32, self.device)
                loss = self.enc.enc_loss(b, latent)
                train_loss = train_loss + loss
              self.log(train_loss_info, loss_infos, '_train')
              self.optim.zero_grad()
              train_loss.backward()
              if self._grad_norm:
                  nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self._grad_norm)
              self.optim.step()
              # torch.cuda.empty_cache()
      return loss_infos

  @torch.no_grad()
  def process_fn(self,batch, buffer, indices):
      if self._recompute_adv:
          # buffer input `buffer` and `indices` to be used in `learn()`.
          self._buffer, self._indices = buffer, indices
      batch = self._compute_returnsV2(batch, buffer, indices)
      batch.act = to_torch_as(batch.act, batch.v_s)
      logp_old = self.dist_fn(batch.policy.probs).log_prob(batch.act)
      batch.logp_old = logp_old
      batch.to_torch(torch.float32, 'cpu')
      return batch
  @torch.no_grad()
  def _compute_returnsV2(self,batch, buffer, indices):
     v_s_ = buffer.get(buffer.next(indices), 'policy', Batch()).fvalues.flatten()
     v_s  = batch.policy.fvalues.flatten()
     batch.v_s = v_s
     batch.v_s_ = v_s_
     if self._rew_norm:  # unnormalize v_s & v_s_
         v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
         v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
     unnormalized_returns, advantages = self.compute_episodic_return(
         batch,
         buffer,
         indices,
         v_s_,
         v_s,
         gamma=self._gamma,
         gae_lambda=self._lambda
     )
     if self._rew_norm:
         batch.returns = unnormalized_returns / \
             np.sqrt(self.ret_rms.var + self._eps)
         self.ret_rms.update(unnormalized_returns)
     else:
         batch.returns = unnormalized_returns
     batch.returns = to_torch_as(batch.returns, batch.v_s)
     batch.adv = to_torch_as(advantages, batch.v_s)
     return batch

from space_wrapper import SpaceWrapper
from relation_net import RNModule, RNEncoder

class PPO(PPOBase):
  def __init__(self, obs_space, act_space, device='cpu', pol_kwargs={}, ppo_kwargs={}):
    self.cfg = pol_kwargs
    super().__init__(obs_space, act_space, device=device,
                     latent_dim=pol_kwargs.get('latent_dim', 256),
                     net_arch=pol_kwargs.get('net_arch', [256,256]),
                     **ppo_kwargs)
    apply_init(self)

class ObjSpacePolicy(PPOBase):
  def __init__(self, obs_space, act_space, device='cpu', pol_kwargs={}, ppo_kwargs={}):
    self.cfg = pol_kwargs
    def make_enc_ac(a, b):
      enc = RNEncoder(obs_space, act_space, cfg=pol_kwargs.encoder)
      ac = RNModule(enc.output_shape, act_space, cfg=pol_kwargs.reasoning_layer)
      return enc, ac
    super().__init__(obs_space, act_space, make_enc_ac=make_enc_ac, device=device, **ppo_kwargs)
    apply_init(self)

from smorl import *
class SMORL(PPOBase):
  def __init__(self, obs_space, act_space, device='cpu', pol_kwargs={}, ppo_kwargs={}):
    self.cfg = pol_kwargs
    def make_enc_ac(a, b):
      enc = SMORLEncoder(self.cfg, self.cfg.input_shape)
      ac = create_mlp(enc.output_dim, act_space.n + 1, [64], return_seq=True)
      return enc, ac
    super().__init__(obs_space, act_space, make_enc_ac=make_enc_ac, device=device, **ppo_kwargs)
    apply_init(self)
