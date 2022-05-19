import gmvae
import utils
import torch
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from fg import ZWhatEnc, GlimpseDec 
import itertools
from torch.distributions import Normal, kl_divergence
import stable_baselines3.common.logger as L
from torch.utils.tensorboard import SummaryWriter
import omegaconf as oc

class GlimpseData(Dataset):
  def __init__(self, path='/lustre/S/yiqi/work/ILP/object/space/more_data_embedding/embedding/embeddings.pkl'): 
    super().__init__()
    self.data = torch.load(path)['glimpse']
  def __getitem__(self, idx):
    return self.data[idx]
  def __len__(self):
    return len(self.data)

class Model(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.enc = ZWhatEnc(cfg)
    self.dec = GlimpseDec(cfg)
    self.to(cfg.device)
  def forward(self, x):
    x = torch.as_tensor(x).float().to(self.cfg.device) 
    z, kl_x, infos = self.enc(x) 
    x_recon_mean, alpha = self.dec(z)
    x_likehood = Normal(x_recon_mean, cfg.fg_sigma).log_prob(x)
    loss = kl_x.mean() + x_likehood.sum([1,2,3]).mean()
    infos.update(dict(x_recon=x_recon_mean.detach().cpu()))
    return loss, infos

def train(cfg):
  dataset = GlimpseData(cfg.data_path)
  dataloader = DataLoader(dataset, cfg.batch_size, shuffle=True)
  model = Model(cfg)
  opt = optim.Adam(model, lr=1e-3) 
  L.configure(cfg.log_dir, ['csv', 'stdout'])
  writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=10)
  st = 0
  for ep in range(cfg.epoch):
    model.train()
    for i, batch in enumerate(dataloader):
      opt.zero_grad()
      loss, infos = model(batch)
      loss.backward()
      opt.step()
      if st % 500 == 0:
        for lname in infos:
          if lname.startswith('gmvae'):
            L.record(lname, infos[lname])
        L.record('step_in_ep', i)
        L.record('Ep', ep)
        L.dump(step=st)
      st += 1
    torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'model.pkl'))
    if ep % 8 == 0 and ep > 0:
      model.eval()
      batch = next(DataLoader(dataset, 2048))
      _, infos = model(batch)
      writer.add_embedding(infos['enc_x'], metadata=infos.get('enc_x_cat', None), label_img=batch, global_step=st)

if __name__ == '__main__':
  cfg = oc.OmegeConf.load('/lustre/S/yiqi/work/ILP/object/space/space/space_small.yaml')
  cfg = oc.OmegeConf.merge(cfg,
                          dict(device='cuda', batch_size=32, log_dir='./log',
                               epoch=100,
                               fg_sigma = 0.15,
                               data_path='/lustre/S/yiqi/work/ILP/object/space/more_data_embedding/embedding/embeddings.pkl'))
  train(cfg)
