import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence
import omegaconf as oc
import functools



def rsample(net, o, reshape_fn=None, std=0):
  out = net(o)
  if reshape_fn is not None:
    out = reshape_fn(out)
  out_mean, out_std = out.chunk(2, -1)
  out_std = F.softplus(out_std) + std
  out_dist = Normal(out_mean, out_std)
  out = out_dist.rsample()
  return out_dist, out

class MultiLinear(nn.Module):
  def __init__(self, in_channels, out_channels, num_linears=2, add_bias=True,):
    super().__init__()
    self.W = nn.Parameter(torch.randn(num_linears, in_channels, out_channels))
    self.b = nn.Parameter(torch.zeros(num_linears, out_channels))
    self.add_bias = add_bias
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.num_linears = num_linears
    nn.init.orthogonal_(self.W, nn.init.calculate_gain('relu'))
  @staticmethod
  def broadcast(x, num_linears):
    shape = list(x.shape)
    extended_shape = shape[:-1] + [num_linears] + shape[-1:] 
    return x.unsqueeze(-2).expand(extended_shape)
  """
    input:  (..., n, cin)
    output: (..., n, cout)
  """
  def forward(self, x):
    x = x.unsqueeze(-2)               # (..., n, 1, in_channels)
    out = torch.matmul(x, self.W)     # (..., n, 1, out_channels)
    out = out.squeeze(-2)
    if self.add_bias:
      out = out + self.b    # (..., n, out_channels)
    return out
  def __repr__(self):
    return f'MultiLinear(in={self.in_channels}, out={self.out_channels}, num={self.num_linears}, add_bias={self.add_bias})'

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
  def forward(self, o, **kwargs):
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

class FlimNet(nn.Module):
  def __init__(self, net, flim_net):
    super().__init__()
    self.net = net
    self.flim_net = flim_net
  def forward(self, x):
    x, fx = x
    flim_iter = 0
    for net in self.net:
      x = net(x) # N C H W
      if isinstance(net, nn.BatchNorm2d):
        flim_net = self.flim_net[flim_iter]
        gamma, beta = flim_net(fx).chunk(2, -1) # N C
        x = x * gamma.reshape(*gamma.shape, 1, 1).expand_as(x) + beta.reshape(*beta.shape, 1,1).expand_as(x)
        flim_iter += 1
    return x

def make_glimpse_enc(cin, cout, use_mlp=False):
    if use_mlp:
      return nn.Sequential(nn.Flatten(), create_mlp(cin * 16 * 16, cout, [512, 512], return_seq=True))
    return nn.Sequential(
              nn.Conv2d(cin, 16, 3, 1, 1),
              nn.CELU(inplace=True),
              nn.GroupNorm(4, 16),
              nn.Conv2d(16, 32, 4, 2, 1),
              nn.CELU(inplace=True),
              nn.GroupNorm(4, 32),
              nn.Conv2d(32, 64, 4, 2, 1),
              nn.CELU(inplace=True),
              nn.GroupNorm(8, 64),
              nn.Conv2d(64, 128, 4),
              nn.CELU(inplace=True),
              nn.GroupNorm(8, 128),
              nn.Flatten(-3),
              nn.Linear(128, cout)
          )

def make_glimpse_dec(z_what_dim, use_mlp=False, cout=3):
    if use_mlp:
      return nn.Sequential(create_mlp(z_what_dim, cout*16*16, [512, 512], return_seq=True), LastDimReshape((cout,16,16)))
    return nn.Sequential(
      nn.Linear(z_what_dim, 128),
      nn.CELU(inplace=True),
      LastDimReshape((128,1,1)),
      nn.ConvTranspose2d(128, 64, 4), 
      nn.CELU(inplace=True),
      nn.GroupNorm(4, 64),
      nn.ConvTranspose2d(64,32,4,2,1), 
      nn.CELU(inplace=True),
      nn.GroupNorm(4, 32),
      nn.ConvTranspose2d(32,16,4,2,1), 
      nn.CELU(inplace=True),
      nn.GroupNorm(4, 16),
      nn.Conv2d(16,cout,3,1,1), 
    )

class SeperateGMVAE(nn.Module):
  def __init__(self, arch):
    super().__init__()
    self.arch = arch
    assert arch.glimpse_size == 16
    self.y_net = make_glimpse_enc(3, arch.num_cat)
    self.use_flim = arch.get('use_flim', False)
    if not self.use_flim:
      self.z_net = make_glimpse_enc(3+arch.num_cat, arch.z_what_dim * 2)
    else:
      z_net = nn.Sequential(
                nn.Conv2d(3 + arch.num_cat, 16, 3, 1, 1),
                nn.CELU(inplace=True),
                nn.BatchNorm2d(16, affine=False),
                nn.Conv2d(16, 32, 4, 2, 1),
                nn.CELU(inplace=True),
                nn.BatchNorm2d(32, affine=False),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.CELU(inplace=True),
                nn.BatchNorm2d(64, affine=False),
                nn.Conv2d(64, 128, 4),
                nn.CELU(inplace=True),
                nn.BatchNorm2d(128, affine=False),
                nn.Flatten(-3),
                nn.Linear(128, arch.z_what_dim * 2)
            )
      flim_net = nn.ModuleList([nn.Linear(arch.num_cat, k * 2) for k in [16, 32, 64, 128]])
      self.z_net = FlimNet(z_net, flim_net)

    self.y_prior = torch.Tensor([1./arch.num_cat] * arch.num_cat)
    self.z_prior_net = nn.Linear(arch.num_cat, 2 * arch.z_what_dim)
  def rsample(self, net, o, reshape_fn=None):
    out = net(o)
    if reshape_fn is not None:
      out = reshape_fn(out)
    out_mean, out_std = out.chunk(2, -1)
    out_std = F.softplus(out_std)
    out_dist = Normal(out_mean, out_std)
    out = out_dist.rsample()
    return out_dist, out
  def forward(self, x, tau=1., **kwargs):
    y = self.y_net(x)
    y_onehot = F.gumbel_softmax(y, tau=tau) # (B, num_cat)
    xy = torch.cat([x, einops.repeat(y_onehot, 'b n -> b n h w', h=16, w=16)], dim=1) # (B, num_cat + input_dim, 16, 16)
    if not self.use_flim:
      z_dist, out  = self.rsample(self.z_net, xy) # (B, output_dim)
    else:
      z_dist, out  = self.rsample(self.z_net, (xy, y_onehot)) # (B, output_dim)
    z_prior_dist, z_prior = self.rsample(self.z_prior_net, y_onehot) # (B, output_dim)
    detach_z_prior = kwargs.get('detach_z_prior', False)
    if detach_z_prior:
      z_prior_dist = Normal(z_prior_dist.loc.detach(), z_prior_dist.scale.detach())
    kl_y = kl_divergence(Categorical(logits=y), Categorical(probs=self.y_prior.to(x.device))) # (B,)
    alpha = self.arch.get('kl_balance_alpha', -1)
    if alpha > 0:
      kl_z = kl_divergence(z_dist, Normal(z_prior_dist.loc.detach(), z_prior_dist.scale.detach())) * alpha +\
             kl_divergence(Normal(z_dist.loc.detach(), z_dist.scale.detach()), z_prior_dist) * (1-alpha)
      kl_z = kl_z.sum(-1)
    else:
      kl_z = kl_divergence(z_dist, z_prior_dist).sum(-1) # (B,)
    total_kl = kl_y + kl_z
    infos = dict(enc_x=out.detach().cpu(), enc_x_cat=F.softmax(y, dim=-1).detach().cpu(), gmvae_kl_y = kl_y.mean().item(), gmvae_kl_z = kl_z.mean().item(), gmvae_kl_total = total_kl.mean().item())
    return out, total_kl, infos

class SeperateGMVAEV2(SeperateGMVAE):
  def forward(self, x, tau=1., sample_zout=True, ret_probs_y=False, **kwargs):
    y = self.y_net(x) # b, n
    probs_y = F.softmax(y, dim=-1)
    eye_y = torch.eye(self.arch.num_cat).to(y.device) # n, n
    eye_y_ = einops.repeat(eye_y, 'n n2 -> (b n) n2 h w', b=x.shape[0], h=x.shape[-2], w=x.shape[-1])
    x_ = einops.repeat(x, 'b c h w -> (b n) c h w', n=self.arch.num_cat)
    stacked_xy = torch.cat([x_, eye_y_], dim=-3) # b, n, n+c, h, w
    z_dist, z_out = self.rsample(self.z_net, stacked_xy,
                          reshape_fn=lambda o: einops.rearrange(o, '(b n) z -> b n z', b=x.shape[0])) # b, n, 2*out_dim

    # compute output.
    if sample_zout:
      y_onehot = F.gumbel_softmax(y, tau=tau) # (b, n), hard=False
      out = (z_out * y_onehot.unsqueeze(-1)).sum(1) # (b, z)
    else:
      out = z_out # (b, n, z)
    # compute kl loss
    z_prior_dist, _ = self.rsample(self.z_prior_net, eye_y) # (n, 2*out_dim)
    # print(z_dist.loc.shape, z_prior_dist.loc.shape)
    kl_z = kl_divergence(z_dist, z_prior_dist).sum(-1) # (b, n) 
    kl_z = (probs_y * kl_z).sum(-1)
    kl_y = kl_divergence(Categorical(probs=probs_y), Categorical(probs=self.y_prior.to(x.device)))
    total_kl = kl_y + kl_z
    infos = dict(enc_x=out.detach().cpu(), enc_x_cat=probs_y.detach().cpu(), gmvae_kl_y = kl_y.mean().item(), gmvae_kl_z = kl_z.mean().item(), gmvae_kl_total = total_kl.mean().item())
    if ret_probs_y:
      return probs_y, out, total_kl, infos
    return out, total_kl, infos

class SeperateGMVAEV3(SeperateGMVAEV2):
  def __init__(self, arch):
    super().__init__(arch)
    self.dec = nn.Sequential(
      nn.Linear(arch.z_what_dim, 128),
      nn.CELU(inplace=True),
      LastDimReshape((128,1,1)),
      nn.ConvTranspose2d(128, 64, 4), 
      nn.CELU(inplace=True),
      nn.GroupNorm(4, 64),
      nn.ConvTranspose2d(64,32,4,2,1), 
      nn.CELU(inplace=True),
      nn.GroupNorm(4, 32),
      nn.ConvTranspose2d(32,16,4,2,1), 
      nn.CELU(inplace=True),
      nn.GroupNorm(4, 16),
      nn.Conv2d(16,3,3,1,1), 
    )
  def forward(self, x, recon=False):
    probs_y, out, total_kl, infos = super().forward(x, sample_zout=False, ret_probs_y=True)
    if recon:
      out = einops.rearrange(out, 'b n z -> (b n) z')
      recon_x = einops.rearrange(self.dec(out), '(b n) c h w -> b n c h w', b=x.shape[0])
      x_ = x.unsqueeze(1)
      loss = F.mse_loss(recon_x, x_, reduction='none').sum([-3,-2,-1]) * probs_y
      loss = loss.sum(1) # (B,)
      return loss, total_kl, infos
    else:
      return out, total_kl, infos

class SeperateGMVAEV4(nn.Module):
  def __init__(self, arch):
    self.arch = arch
    super().__init__()
    c = 3 if not arch.get('use_bce', False) else 1
    self.use_bce = arch.get('use_bce', False)
    # self.y_net = make_glimpse_enc(c, arch.num_cat, True)
    # self.z_net = create_mlp(c * 16 * 16 + arch.num_cat, arch.z_what_dim * 2, [512], return_seq=True)
    self.enc = make_glimpse_enc(c, 128, arch.get('sep_use_mlp', False))
    self.y_net = create_mlp(128, arch.num_cat, [64], return_seq=True)
    self.z_net = create_mlp(128, arch.z_what_dim * 2, [64], return_seq=True,
                           create_layer=functools.partial(MultiLinear, num_linears=arch.num_cat))
    self.dec = make_glimpse_dec(arch.z_what_dim, arch.get('sep_use_mlp', False),cout=c) 
    self.y_prior = torch.Tensor([1./arch.num_cat] * arch.num_cat)
    self.z_prior_net = nn.Linear(arch.num_cat, 2 * arch.z_what_dim)

  def forward(self, x, recon=False, std=0):
    latent = self.enc(x)
    y = self.y_net(latent)
    probs_y = F.softmax(y, dim=-1)
    latent_bc = MultiLinear.broadcast(latent, self.arch.num_cat) 
    z_dist, out = rsample(self.z_net, latent_bc) # (b, n, 2*out_dim)
    # y = self.y_net(x)
    # probs_y = F.softmax(y, dim=-1)
    # z_net_in = torch.cat([x.flatten(1).unsqueeze(-2).expand(-1, self.arch.num_cat, -1),
    #                       torch.eye(self.arch.num_cat).to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1)],
    #                      dim = -1)
    # z_dist, out = rsample(self.z_net, z_net_in)

    eye_y = torch.eye(self.arch.num_cat).to(y.device) # n, n
    z_prior_dist, _ = rsample(self.z_prior_net, eye_y, std=std) # (n, 2*out_dim)

    kl_z = kl_divergence(z_dist, z_prior_dist).sum(-1) # (b, n) 
    kl_z = (kl_z * probs_y).sum(-1)
    kl_y = kl_divergence(Categorical(probs=probs_y), Categorical(probs=self.y_prior.to(x.device))) # (b,)
    total_kl = kl_y + kl_z
    infos = dict(enc_x=out.detach().cpu(), enc_x_cat=probs_y.detach().cpu(),
                  gmvae_y_ent = Categorical(probs=probs_y).entropy().mean().item(),
                  gmvae_kl_y = kl_y.mean().item(),
                  gmvae_kl_z = kl_z.mean().item(),
                  gmvae_kl_total = total_kl.mean().item())
    if not recon:
      return out, total_kl, infos
    else:
      out = einops.rearrange(out, 'b n z -> (b n) z')
      recon_x = einops.rearrange(self.dec(out), '(b n) c h w -> b n c h w', b=x.shape[0])
      x_ = x.unsqueeze(1).expand_as(recon_x)
      if self.use_bce:
        recon_x = F.sigmoid(recon_x)
        loss = torch.nn.BCELoss(reduction='none')(recon_x, x_).sum([-3,-2,-1]) * probs_y
      else:
        loss = F.mse_loss(recon_x, x_, reduction='none').sum([-3,-2,-1]) * probs_y
      loss = loss.sum(1) # (B,)
      infos['gmvae_recon_loss'] = loss.mean().item()
      infos['recon_x'] = recon_x.detach().cpu()
      return loss, total_kl, infos

class SeperateGMVAEV5(nn.Module):
  def __init__(self, arch):
    self.arch = arch
    super().__init__()
    self.enc = make_glimpse_enc(3, 128, arch.get('sep_use_mlp', False))
    self.y_net = create_mlp(128, arch.num_cat, [64], return_seq=True)
    self.z_net = create_mlp(128, arch.z_what_dim * 2, [64], return_seq=True,
                           create_layer=functools.partial(MultiLinear, num_linears=arch.num_cat))

    self.dec = nn.Sequential(create_mlp(arch.z_what_dim, 3*16*16, [256, 256], create_layer=functools.partial(MultiLinear, num_linears=arch.num_cat), return_seq=True), LastDimReshape((3,16,16)))
    self.y_prior = torch.Tensor([1./arch.num_cat] * arch.num_cat)
    self.z_prior_net = nn.Linear(arch.num_cat, 2 * arch.z_what_dim)

  def forward(self, x, recon=False):
    latent = self.enc(x)
    y = self.y_net(latent)
    probs_y = F.softmax(y, dim=-1)
    latent_bc = MultiLinear.broadcast(latent, self.arch.num_cat) 
    z_dist, out = rsample(self.z_net, latent_bc) # (b, n, 2*out_dim)

    eye_y = torch.eye(self.arch.num_cat).to(y.device) # n, n
    z_prior_dist, _ = rsample(self.z_prior_net, eye_y) # (n, 2*out_dim)

    kl_z = kl_divergence(z_dist, z_prior_dist).sum(-1) # (b, n) 
    kl_z = (kl_z * probs_y).sum(-1)
    kl_y = kl_divergence(Categorical(probs=probs_y), Categorical(probs=self.y_prior.to(x.device))) # (b,)

    total_kl = kl_y + kl_z
    infos = dict(enc_x=out.detach().cpu(), enc_x_cat=probs_y.detach().cpu(), gmvae_kl_y = kl_y.mean().item(), gmvae_kl_z = kl_z.mean().item(), gmvae_kl_total = total_kl.mean().item())
    if not recon:
      return out, total_kl, infos
    else:
      # out = einops.rearrange(out, 'b n z -> (b n) z')
      # recon_x = einops.rearrange(self.dec(out), '(b n) c h w -> b n c h w', b=x.shape[0])
      recon_x = self.dec(out)
      x_ = x.unsqueeze(1)
      loss = F.mse_loss(recon_x, x_, reduction='none').sum([-3,-2,-1]) * probs_y
      loss = loss.sum(1) # (B,)
      infos['gmvae_recon_loss'] = loss.mean().item()
      infos['recon_x'] = recon_x.detach().cpu()
      return loss, total_kl, infos


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
    if len(arch.get('y_net_arch', [])) == 0:
      self.y_net = nn.Linear(input_dim, arch.num_cat)
    else:
      self.y_net = create_mlp(input_dim, arch.num_cat, arch.get('y_net_arch', []), return_seq=True)

    # self.y_net = create_mlp(input_dim, arch.num_cat, arch.get('y_net_arch', []), return_seq=True)

    self.z_net = create_mlp(input_dim + arch.num_cat, 2 * output_dim, arch.get('z_net_arch', [output_dim]), return_seq=True)
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
  def forward(self, x, tau=1., **kwargs):
    y = self.y_net(x)
    y_onehot = F.gumbel_softmax(y, tau=tau) # (B, num_cat)
    xy = torch.cat([x, y_onehot], dim=-1) # (B, num_cat + input_dim)
    z_dist, z  = self.rsample(self.z_net, xy) # (B, output_dim)
    detach_z_prior = kwargs.get('detach_z_prior', False)
    z_prior_dist, z_prior = self.rsample(self.z_prior_net, y_onehot) # (B, output_dim)
    if detach_z_prior:
      z_prior_dist = Normal(z_prior_dist.loc.detach(), z_prior_dist.scale.detach())
    # out = z 
    out = self.out_net(torch.cat([z, y_onehot], dim=-1))
    kl_y = kl_divergence(Categorical(logits=y), Categorical(probs=self.y_prior.to(x.device))) # (B,)
    kl_z = kl_divergence(z_dist, z_prior_dist).sum(-1) # (B,)
    total_kl = kl_y + kl_z
    infos = dict(enc_x=out.detach().cpu(), enc_x_cat=F.softmax(y, dim=-1).detach().cpu(), gmvae_kl_y = kl_y.mean().item(), gmvae_kl_z = kl_z.mean().item(), gmvae_kl_total = total_kl.mean().item())
    return out, total_kl, infos


class SimpleGMVAEV2(nn.Module):
  """
  According to https://arxiv.org/pdf/1611.05148.pdf.
  Inference Model: q(z,c|x) = q(z|x)q(c|x), where q(c|x) is estimated using E_z q(c|z)
  Generate Model:  p(x,z,c) = p(c)p(z|c)p(x|z)
  Note: q(z|x) shares embedding o = o(x), and p(x|z) is not included in this module.
  """
  def __init__(self, input_dim, output_dim, arch):
    super().__init__()
    self.arch = arch
    n_classes = arch.num_cat
    self._pi = nn.Parameter(torch.zeros(n_classes))
    self.mu_c = nn.Parameter(torch.randn(n_classes, output_dim))
    self.logvar_c = nn.Parameter(torch.randn(n_classes, output_dim))
    self.z_net = create_mlp(input_dim, output_dim * 2, [], return_seq=True)

  @property
  def prior_z(self):
    return torch.cat([self.mu_c, self.logvar_c], dim=-1)
  # @property
  # def pi(self):
  #   return F.softmax(self._pi, dim=-1)
  def rsample(self, net, o):
    out_mean, out_std = net(o).chunk(2, -1)
    out_std = F.softplus(out_std)
    out_dist = Normal(out_mean, out_std)
    out = out_dist.rsample()
    return out_dist, out

  def compute_loss(self, z_dist):
    # TODO: fix prior.
    _pi = self._pi * 0
    # Compute gamma ( q(c|x) )
    z0 = z_dist.rsample().unsqueeze(1) # B, 1, Z


    p_z_c_dist, _ = self.rsample(nn.Identity(), self.prior_z) # (N, Z)

    if False:
      p_z0_c = p_z_c_dist.log_prob(z0).sum(-1).exp() # B, N
      p_c_z0 = F.softmax(_pi, dim=-1) * p_z0_c
      p_c_z0 = p_c_z0 / p_c_z0.sum(-1, keepdim=True)
    else:
      p_c_z0 = F.softmax(p_z_c_dist.log_prob(z0).sum(-1) + _pi, dim=-1)# B, N

    p_c_x = p_c_z0

    kl_z = p_c_x * \
            kl_divergence(\
              Normal(z_dist.loc.unsqueeze(1), z_dist.scale.unsqueeze(1)), \
              p_z_c_dist).sum(-1)  # (B, N)
    kl_z = kl_z.sum(-1) # (B,)

    kl_c = kl_divergence(Categorical(probs=p_c_x), Categorical(probs=F.softmax(_pi, dim=-1))) # (B,)
    total_kl = kl_z + kl_c
    return p_c_x, total_kl, kl_z, kl_c
    # h = z - self.mu_c # B, N, Z
    # h = torch.exp(-0.5 * torch.sum((h * h / self.logvar_c.exp()), dim=2)) # B, N
    # Same as `torch.sqrt(torch.prod(model.logvar.exp(), dim=1))`
    # h = h / torch.sum(0.5 * self.logvar_c, dim=1).exp() # B, N
    # p_z_given_c = h / (2 * math.pi)
    # p_z_c = p_z_given_c * self.weights
    # gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True) # B, N

    # h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu).pow(2)
    # h = torch.sum(model.logvar + h / model.logvar.exp(), dim=2)
    # loss = F.binary_cross_entropy(recon_x, x, reduction='sum') \
    #     + 0.5 * torch.sum(gamma * h) \
    #     - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
    #     + torch.sum(gamma * torch.log(gamma + 1e-9)) \
    #     - 0.5 * torch.sum(1 + logvar)
  def forward(self, x, tau=0, ret_mean=False, **kwargs):
    z_dist, z = self.rsample(self.z_net, x)
    if ret_mean:
      z = z_dist.loc
    z_cat, total_kl, kl_z, kl_c = self.compute_loss(z_dist)
    return z, total_kl, dict(enc_x=z.detach().cpu(), enc_x_cat=z_cat.detach().cpu(),\
                             gmvae_kl_c=kl_c.mean().item(),\
                             gmvae_kl_z=kl_z.mean().item(),\
                             gmvae_kl_total=total_kl.mean().item())
    


if __name__ == '__main__':
  arch = oc.DictConfig(dict(num_cat=10, glimpse_size=16, z_what_dim=32, use_flim=False))
  gmvae = SeperateGMVAEV4(arch)
  inputs = torch.randn(16,3,16,16)
  out, kl, infos = gmvae(inputs, recon=True)
  print(out.shape, kl.mean(), infos)
