import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence
import omegaconf as oc
import numpy as np
import functools
from torchvision.utils import make_grid
import gmvae
import utils
import torch
import os
from torch import nn, optim
from torch.utils.data import TensorDataset,Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import itertools
from torch.distributions import Normal, kl_divergence
import stable_baselines3.common.logger as L
from torch.utils.tensorboard import SummaryWriter
import omegaconf as oc
from gmvae import *

import torchvision.transforms.functional as VF
from scipy.stats import mode

use_bce = False
use_minst = False



def loadMNIST():
  mnist =MNIST(root='/lustre/S/yiqi/work/ILP/object/torch_dataset/mnist', train=True, download=True)
  data = mnist.data
  data = data.unsqueeze(1).expand(-1, 3 if not use_bce else 1, -1, -1)
  data = VF.resize(data, 16).float() / 255.
  mnist.data = data
  return mnist, data
mnist, data = loadMNIST()
data = torch.load('/lustre/S/yiqi/work/ILP/object/obj_rn2/datasets/all_glimpse.pt')
print(data.shape)
dataloader = DataLoader(TensorDataset(data), batch_size=1024)
device = 'cuda'
def test_acc(model):
    if not use_minst:
      return -1
    imgs = mnist.data[0:10000].to(device)
    labels = mnist.targets[0:10000] 
    probs = model(imgs, recon=False)[-1]['enc_x_cat'] 
    cat_pred = probs.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    for cat in range(probs.shape[1]):
        idx = cat_pred == cat
        lab = labels[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    real_pred = np.asarray(real_pred).astype(np.int32)
    labels = np.asarray(labels).astype(np.int32)
    return np.mean(real_pred == labels)

seed = np.random.randint(1, 10000000)
arch = oc.DictConfig(dict(epoch=1000, num_cat=10 if use_minst else 4, z_what_dim=64, sep_use_mlp=True, use_bce=use_bce))
arch.log_dir=f'log/MINST{int(use_minst)}_Z{arch.z_what_dim}_MLP{int(arch.sep_use_mlp)}_seed{seed}'
os.makedirs(arch.log_dir, exist_ok=True)
model = SeperateGMVAEV4(arch).to(device)
# colors = torch.Tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0]])
colors = [[255,0,0], [255,255,0], [255,153,18],[255,127,80],[255,192,203],[255,0,255], [0,255,0], [0,255,255], [8,46,84], [0,199,140], [0,0,255], [160,32,240],[218,112,214]] 
colors = torch.Tensor(colors) / 255.
N = 64

class QVAE(nn.Module):
  def __init__(self, arch):
    super().__init__()
    self.arch = arch
    self.enc = make_glimpse_enc(3, arch.num_cat, True)
    self.dec = make_glimpse_dec(arch.num_cat, True)
    self.zero = torch.Tensor([0]).mean()
    
  def forward(self, x, **kwargs):
    z = self.enc(x)
    q = F.gumbel_softmax(z, dim=-1)
    recon_x = self.dec(q)
    loss = F.mse_loss(recon_x, x, reduction='none').sum([1,2,3]).mean()
    recon_x = einops.repeat(self.dec(torch.eye(self.arch.num_cat).to(q.device)).detach().cpu(), 'n c h w -> b n c h w', b=x.shape[0])
    infos = dict(recon_x = recon_x, enc_x=self.zero, enc_x_cat=F.softmax(q, dim=-1).detach().cpu())
    return loss, self.zero, infos

def train(cfg):
  opt = optim.Adam(model.parameters(), lr=1e-3) 
  L.configure(cfg.log_dir, ['csv', 'stdout'])
  writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=10)
  st = 0
  for ep in range(cfg.epoch):
    model.train()
    for i, batch in enumerate(dataloader):
      batch = batch[0].to(device)
      opt.zero_grad()
      loss, kl, infos = model(batch, recon=True)
      loss = (loss + kl).mean()
      loss.backward()
      opt.step()
      enc_cat_mean = infos['enc_x_cat'].mean(0) # (N,)
      for i in range(cfg.num_cat):
        writer.add_scalar(f'CatProb/Cat{i}', enc_cat_mean[i], st)
        L.record_mean(f'CatProb/Cat{i}', enc_cat_mean[i])
      for k in infos:
        if k.startswith('gmvae'):
          writer.add_scalar(f'gmvae/{k}', infos[k], global_step=st)
          L.record_mean(k, infos[k])
      if st % 100 == 0:
        acc = test_acc(model)
        writer.add_scalar('ACC', acc, st)
        L.record('ACC', acc)
        img = batch[:N].expand(N, 3, 16, 16).clone()
        cat = infos['enc_x_cat'][:N].argmax(-1)
        img[:,:,:3] = colors[cat].reshape(-1,3,1,1)
        grid = make_grid(img, 8) 
        writer.add_image('CatInfo', grid, st)
        if True:
          for i in range(cfg.num_cat):
            recon_xi = infos['recon_x'][:N, i]
            grid = make_grid(recon_xi, 8)
            writer.add_image(f'ReconX/Cat{i}', grid, global_step=st)

        L.record('Step', st)
        L.record('Ep', ep)
        L.dump(step=st)
      st += 1
      print(f'Epoch: {ep}, Step: {st}', end='\r')
    torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'model.pkl'))

if __name__ == '__main__':
  cfg = arch
  train(cfg)
