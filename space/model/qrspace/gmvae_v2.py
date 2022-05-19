import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence
import omegaconf as oc

def create_mlp(
    input_dim,
    output_dim,
    net_arch,
    activation_fn= nn.ReLU,
    squash_output=False,
    create_layer=nn.Linear,
    return_seq =False,
):
    if len(net_arch) > 0:
        modules = [create_layer(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(create_layer(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(create_layer(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    if return_seq:
      modules = nn.Sequential(*modules)
    return modules

class LastDimReshape(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.shape = shape
  def forward(self, o):
    return o.reshape(*o.shape[:-1], *self.shape)

class VAE(nn.Module):
  def __init__(self, input_dim, output_dim, arch):
    super().__init__()
    self.arch = arch
    self.net = nn.Linear(input_dim, output_dim * 2)
  def rsample(self, net, o):
    out_mean, out_std = net(o).chunk(2, -1)
    out_std = F.softplus(out_std)
    out_dist = Normal(out_mean, out_std)
    out = out_dist.rsample()
    return out_dist, out
  def forward(self, o):
    out_dist, out = self.rsample(self.net, o)
    total_kl = kl_divergence(out_dist, Normal(0,1)).sum(-1)
    infos = dict(enc_x = out.detach().cpu(), enc_x_cat=None, gmvae_kl_total =total_kl.mean().item())
    return out, total_kl, infos

class GMVAE(nn.Module):
  """
    arch: num_cat, M
  """
  def __init__(self, input_dim, output_dim, arch):
    super().__init__()
    self.arch = arch
    num_cat = arch.num_cat
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.x_net = nn.Linear(input_dim, 2 * output_dim)
    self.w_net = nn.Linear(input_dim, 2 * output_dim)
    self.z_net = nn.Linear(output_dim * 2, num_cat) 
    self.x_prior_net = nn.Sequential(nn.Linear(output_dim, 2 * num_cat * output_dim), LastDimReshape(shape=(num_cat, 2 * output_dim)))
    self.z_net_prior = torch.Tensor([1./num_cat] * num_cat)
    self.w_prior_dist = Normal(0, 1)

  def rsample(self, net, o):
    out_mean, out_std = net(o).chunk(2, -1)
    out_std = F.softplus(out_std)
    out_dist = Normal(out_mean, out_std)
    out = out_dist.rsample()
    return out_dist, out

  """
    o: (B, input_dim)
  """
  def forward(self, o):
    x_dist, x = self.rsample(self.x_net, o)  # TODO: (B, output_dim)
    w_dist, w = self.rsample(self.w_net, o)  # (B, output_dim)

    kl_z = 0

    kl_x = 0
    for i in range(self.arch.M):
      w = w_dist.rsample()
      z_probs = F.softmax(self.z_net(torch.cat([x, w], -1)), dim=-1) # (B, num_cat)
      kl_zi = kl_divergence(Categorical(probs=z_probs), Categorical(probs=self.z_net_prior.to(z_probs.device))) # (B, )
      kl_z = kl_z + kl_zi
      x_prior_dist, x_prior = self.rsample(self.x_prior_net, w)  # (B, num_cat, output_dim)
      x_dist_extended = Normal(x_dist.mean.unsqueeze(-2), x_dist.stddev.unsqueeze(-2)) # (B, 1, output_dim)
      kl_xi = kl_divergence(x_dist_extended, x_prior_dist).sum(-1) # (B, num_cat) 
      kl_xi = (z_probs * kl_xi).sum(-1) # (B,)
      kl_x  = kl_xi + kl_x
    kl_x = kl_x / self.arch.M
    kl_z = kl_z / self.arch.M
    kl_z = kl_z.clamp(self.arch.z_lambda, 1000000)
    
    kl_w = kl_divergence(w_dist, self.w_prior_dist).sum(-1) # (B,) 
    # kl_z, kl_x, kl_w = kl_z.mean(), kl_x.mean(), kl_w.mean()

    total_kl = kl_z + kl_x + kl_w

    infos = dict(enc_x=x.detach().cpu(), enc_x_cat=z_probs.detach().cpu(), gmvae_kl_z=kl_z.mean().item(), gmvae_kl_x = kl_x.mean().item(), gmvae_kl_w=kl_w.mean().item(), gmvae_kl_total=total_kl.mean().item())
    return x, total_kl, infos



class SimpleGMVAE(nn.Module):
  """
  Inference Model: q(z,y|x) = q(y|x)q(z|y,x)
  Generate Model:  p(x,z,y) = p(y)p(z|y)p(x|z)
  Note: q(y|x), q(z|y,x) shares embedding o = o(x), and p(x|z) is not included in this module.
  """
  def __init__(self, input_dim, output_dim, arch):
    super().__init__()
    self.arch = arch
    # inference model:
    self.y_net = nn.Linear(input_dim, arch.num_cat)
    self.z_net = create_mlp(input_dim + arch.num_cat, 2 * output_dim, [output_dim], return_seq=True)
    # generative model (without p(x|z)):
    self.y_prior = torch.Tensor([1./arch.num_cat] * arch.num_cat)
    self.z_prior_net = nn.Linear(arch.num_cat, 2 * output_dim)
    self.out_net = nn.Linear(output_dim + arch.num_cat, output_dim)
  def rsample(self, net, o):
    out_mean, out_std = net(o).chunk(2, -1)
    out_std = F.softplus(out_std)
    out_dist = Normal(out_mean, out_std)
    out = out_dist.rsample()
    return out_dist, out
  def forward(self, x, tau=1.):
    y = self.y_net(x)
    y_onehot = F.gumbel_softmax(y, tau=tau) # (B, num_cat)
    
    xy = torch.cat([x, y_onehot], dim=-1) # (B, num_cat + input_dim)
    z_dist, z  = self.rsample(self.z_net, xy) # (B, output_dim)
    z_prior_dist, z_prior = self.rsample(self.z_prior_net, y_onehot) # (B, output_dim)


    # out = z 
    out = self.out_net(torch.cat([z, y_onehot], dim=-1))
    
    kl_y = kl_divergence(Categorical(logits=y), Categorical(probs=self.y_prior.to(x.device))) # (B,)
    kl_z = kl_divergence(z_dist, z_prior_dist).sum(-1) # (B,)
    # kl_y = kl_y.mean()
    # kl_z = kl_z.mean()
  
    total_kl = kl_y + kl_z

    infos = dict(enc_x=out.detach().cpu(), enc_x_cat=F.softmax(y, dim=-1).detach().cpu(), gmvae_kl_y = kl_y.mean().item(), gmvae_kl_z = kl_z.mean().item(), gmvae_kl_total = total_kl.mean().item())

    return out, total_kl, infos

if __name__ == '__main__':
  arch = oc.DictConfig(dict(num_cat=10, z_lambda=0.1, M=10))
  gmvae = GMVAE(32, 16, arch)
  inputs = torch.randn(4,32)
  out, kl, infos = gmvae(inputs)
  print(out.shape, infos)
    




    


