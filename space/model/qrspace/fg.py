import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from .utils import NumericalRelaxedBernoulli, kl_divergence_bern_bern, get_boundary_kernel_new, get_boundary_kernel
from .utils import spatial_transform, linear_annealing
from .gmvae import *
# from .arch import arch

class SpaceFg(nn.Module):
    def __init__(self, arch):
        nn.Module.__init__(self)
        self.arch = arch
        self.img_encoder = ImgEncoderFg(arch)
        if arch.get('z_what_seperate_gmvae', False):
          self.z_what_net = SeperateGMVAE(arch)

        elif arch.get('z_what_seperate_gmvae_v2', False):
          self.z_what_net = SeperateGMVAEV2(arch)
        else:
          self.z_what_net = ZWhatEnc(arch)

        if arch.get('seperate_z_cat_net', False):
          self.z_cat_net = SeperateGMVAEV3(arch)
        elif arch.get('seperate_z_cat_net_v2', False):
          self.z_cat_net = SeperateGMVAEV4(arch)
        else:
          self.z_cat_net = None

        self.glimpse_dec = GlimpseDec(arch)
        # This is what is really used
        self.boundary_kernel = get_boundary_kernel_new(kernel_size=32, boundary_width=6)
        
        self.fg_sigma = arch.fg_sigma
        # I register many things as buffer but this is not really necessary.
        # Temperature for gumbel-softmax
        self.register_buffer('tau', torch.tensor(arch.tau_start_value))
        
        # Priors
        # Some are buffer and some are not. This is stupid. But for compatibility I have to do so.
        self.register_buffer('prior_z_pres_prob', torch.tensor(arch.z_pres_start_value))
        self.register_buffer('prior_what_mean', torch.zeros(1))
        self.register_buffer('prior_what_std', torch.ones(1))
        self.register_buffer('prior_depth_mean', torch.zeros(1))
        self.register_buffer('prior_depth_std', torch.ones(1))
        self.prior_scale_mean_new = torch.tensor(arch.z_scale_mean_start_value)
        self.prior_scale_std_new = torch.tensor(arch.z_scale_std_value)
        self.prior_shift_mean_new = torch.tensor(0.)
        self.prior_shift_std_new = torch.tensor(1.)
        # self.register_buffer('prior_scale_mean_new', torch.tensor(arch.z_scale_mean_start_value))
        # self.register_buffer('prior_scale_std_new', torch.tensor(arch.z_scale_std_value))
        # self.register_buffer('prior_shift_mean_new', torch.tensor(0.))
        # self.register_buffer('prior_shift_std_new', torch.tensor(1.))
        
        # TODO: These are placeholders for loading old checkpoints. No longer used
        self.boundary_filter = get_boundary_kernel(sigma=20)
        self.register_buffer('prior_scale_mean',
                             torch.tensor([arch.z_scale_mean_start_value] * 2).view((arch.z_where_scale_dim), 1, 1))
        self.register_buffer('prior_scale_std',
                             torch.tensor([arch.z_scale_std_value] * 2).view((arch.z_where_scale_dim), 1, 1))
        self.register_buffer('prior_shift_mean',
                             torch.tensor([0., 0.]).view((arch.z_where_shift_dim), 1, 1))
        self.register_buffer('prior_shift_std',
                             torch.tensor([1., 1.]).view((arch.z_where_shift_dim), 1, 1))
    
    @property
    def z_what_prior(self):
        return Normal(self.prior_what_mean, self.prior_what_std)
    
    @property
    def z_depth_prior(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)
    
    @property
    def z_scale_prior(self):
        return Normal(self.prior_scale_mean_new, self.prior_scale_std_new)
    
    @property
    def z_shift_prior(self):
        return Normal(self.prior_shift_mean_new, self.prior_shift_std_new)
    
    def anneal(self, global_step):
        """
        Update everything

        :param global_step: global step (training)
        :return:
        """

        self.prior_z_pres_prob = linear_annealing(self.prior_z_pres_prob.device, global_step,
                                                  self.arch.z_pres_start_step, self.arch.z_pres_end_step,
                                                  self.arch.z_pres_start_value, self.arch.z_pres_end_value)
        self.prior_scale_mean_new = linear_annealing(self.prior_z_pres_prob.device, global_step,
                                                self.arch.z_scale_mean_start_step, self.arch.z_scale_mean_end_step,
                                                self.arch.z_scale_mean_start_value, self.arch.z_scale_mean_end_value)
        self.tau = linear_annealing(self.tau.device, global_step,
                                    self.arch.tau_start_step, self.arch.tau_end_step,
                                    self.arch.tau_start_value, self.arch.tau_end_value)
    
    def forward(self, x, globel_step, encode_only=False, glimpse_only=False):
        """
        Forward pass
        :param x: (B, 3, H, W)
        :param globel_step: global step (training)
        :return:
            fg_likelihood: (B, 3, H, W)
            y_nobg: (B, 3, H, W), foreground reconstruction
            alpha_map: (B, 1, H, W)
            kl: (B,) total foreground kl
            boundary_loss: (B,)
            log: a dictionary containing anything we need for visualization
        """
        B = x.size(0)
        # if globel_step:
        self.anneal(globel_step)
        
        # Everything is (B, G*G, D), where D varies
        z_pres, z_depth, z_scale, z_shift, z_where, \
        z_pres_logits, z_depth_post, z_scale_post, z_shift_post = self.img_encoder(x, self.tau)
        
        # (B, 3, H, W) -> (B*G*G, 3, H, W). Note we must use repeat_interleave instead of repeat
        x_repeat = torch.repeat_interleave(x, self.arch.G ** 2, dim=0)
        
        # (B*G*G, 3, H, W), where G is the grid size
        # Extract glimpse
        x_att = spatial_transform(x_repeat, z_where.view(B * self.arch.G ** 2, 4),
                                  (B * self.arch.G ** 2, 3, self.arch.glimpse_size, self.arch.glimpse_size), inverse=False)
        if glimpse_only:
          # z_pres: (B*G*G), z_shift: (B*G*G, D), x_att: (B*G*G, 3, H, W)
          return dict(z_pres=z_pres.flatten(),
                      glimpse=x_att.detach(),
                      z_where=z_where.reshape(-1, 4).detach(),
                      z_shift=z_shift.reshape(-1, 2))
        
        # (B*G*G, D), (B * G * G,)
        pretrain_stage = globel_step < self.arch.get('freeze_kl_zwhat_step', 0)
        z_what, kl_z_what, z_what_infos = self.z_what_net(x_att, detach_z_prior=pretrain_stage, tau=self.tau)

        sep_total_loss = 0
        if self.z_cat_net:
          z_what_infos = {}
          if not pretrain_stage or encode_only:
            if encode_only:
              obj_att = x_att.detach()
            else:
              obj_att = x_att.detach()[z_pres.flatten() > 0.5]
            if len(obj_att) > 0:
              recon_loss, kl_loss, infos = self.z_cat_net(obj_att.detach(), recon=True)
              sep_total_loss = (recon_loss + kl_loss).mean()
              z_what_infos.update(infos)
        # if pretrain_stage:
        #   kl_z_what = kl_z_what.detach() 
        # if pretrain_stage:
        #   kl_z_what = kl_z_what * 0

        if encode_only:
          if 'enc_x_cat' in z_what_infos:
            z_what_cat = z_what_infos['enc_x_cat'].view(B, self.arch.G**2, -1).to(z_what.device)
          else:
            z_what_cat = None
          return {'z_what': z_what.view(B, self.arch.G ** 2, self.arch.z_what_dim),
                  'z_what_cat': z_what_cat,
                  'z_where': z_where,
                  'z_pres': z_pres,
                  'z_scale': z_scale,
                  'z_shift': z_shift}
        
        # Decode z_what into small reconstructed glimpses
        # All (B*G*G, 3, H, W)
        o_att, alpha_att = self.glimpse_dec(z_what)
        # z_pres: (B, G*G, 1) -> (B*G*G, 1, 1, 1)
        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        # (B*G*G, 3, H, W)
        y_att = alpha_att_hat * o_att
        
        # Compute pixel-wise object weights
        # (B*G*G, 1, H, W). These are glimpse size
        importance_map = alpha_att_hat * 100.0 * torch.sigmoid(-z_depth.view(B*self.arch.G**2, 1, 1, 1))
        # (B*G*G, 1, H, W). These are of full resolution
        importance_map_full_res = spatial_transform(importance_map, z_where.view(B * self.arch.G ** 2, 4), (B*self.arch.G**2, 1, *self.arch.img_shape),
                                                    inverse=True)
        
        # (B*G*G, 1, H, W) -> (B, G*G, 1, H, W)
        importance_map_full_res = importance_map_full_res.view(B, self.arch.G ** 2, 1, *self.arch.img_shape)
        # Normalize (B, >G*G<, 1, H, W)
        importance_map_full_res_norm = torch.softmax(importance_map_full_res, dim=1)
        
        # To full resolution
        # (B*G*G, 3, H, W) -> (B, G*G, 3, H, W)
        y_each_cell = spatial_transform(y_att, z_where.view(B * self.arch.G ** 2, 4), (B * self.arch.G ** 2, 3, *self.arch.img_shape),
                                        inverse=True).view(B, self.arch.G ** 2, 3, *self.arch.img_shape)
        # Weighted sum, (B, 3, H, W)
        y_nobg = (y_each_cell * importance_map_full_res_norm).sum(dim=1)
        
        # To full resolution
        # (B*G*G, 1, H, W) -> (B, G*G, 1, H, W)
        alpha_map = spatial_transform(alpha_att_hat, z_where.view(B * self.arch.G ** 2, 4), (B * self.arch.G ** 2, 1, *self.arch.img_shape),
                                      inverse=True).view(B, self.arch.G ** 2, 1, *self.arch.img_shape)
        
        # Weighted sum, (B, 1, H, W)
        alpha_map = (alpha_map * importance_map_full_res_norm).sum(dim=1)
        
        # Everything is computed. Now let's compute loss
        # Compute KL divergences
        # (B, G*G, 1)
        kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)
        
        # (B, G*G, 1)
        kl_z_depth = kl_divergence(z_depth_post, self.z_depth_prior)
        
        # (B, G*G, 2)
        kl_z_scale = kl_divergence(z_scale_post, self.z_scale_prior)
        kl_z_shift = kl_divergence(z_shift_post, self.z_shift_prior)
        
        # Reshape z_what and z_what_post
        # # (B*G*G, D) -> (B, G*G, D)
        z_what = z_what.view(B, self.arch.G ** 2, self.arch.z_what_dim)
        # z_what_post = Normal(*[x.view(B, self.arch.G ** 2, self.arch.z_what_dim)
        #                        for x in [z_what_post.mean, z_what_post.stddev]])
        # # (B, G*G, D)
        # kl_z_what = kl_divergence(z_what_post, self.z_what_prior)
        if self.arch.get('use_z_pres', False):
          kl_z_what = kl_z_what * z_pres.flatten().detach() 
        kl_z_what = kl_z_what.view(B, self.arch.G**2)
        
        # dimensionality check
        assert ((kl_z_pres.size() == (B, self.arch.G ** 2, 1)) and
                (kl_z_depth.size() == (B, self.arch.G ** 2, 1)) and
                (kl_z_scale.size() == (B, self.arch.G ** 2, 2)) and
                (kl_z_shift.size() == (B, self.arch.G ** 2, 2))
                )
        
        # Reduce (B, G*G, D) -> (B,)
        kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what = [
            x.flatten(start_dim=1).sum(1) for x in [kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what]
        ]
        # (B,)
        kl_z_where = kl_z_scale + kl_z_shift
        
        # Compute boundary loss
        # (1, 1, K, K)
        boundary_kernel = self.boundary_kernel[None, None].to(x.device)
        # (1, 1, K, K) * (B*G*G, 1, 1) -> (B*G*G, 1, K, K)
        boundary_kernel = boundary_kernel * z_pres.view(B * self.arch.G ** 2, 1, 1, 1)
        # (B, G*G, 1, H, W), to full resolution
        boundary_map = spatial_transform(boundary_kernel, z_where.view(B * self.arch.G ** 2, 4), (B * self.arch.G ** 2, 1, *self.arch.img_shape),
                                         inverse=True).view(B, self.arch.G ** 2, 1, *self.arch.img_shape)
        # (B, 1, H, W)
        boundary_map = boundary_map.sum(dim=1)
        # TODO: some magic number. For reproducibility I will keep it
        boundary_map = boundary_map * 1000
        # (B, 1, H, W) * (B, 1, H, W)
        overlap = boundary_map * alpha_map
        # TODO: another magic number. For reproducibility I will keep it
        p_boundary = Normal(0, 0.7)
        # (B, 1, H, W)
        boundary_loss = p_boundary.log_prob(overlap)
        # (B,)
        boundary_loss = boundary_loss.flatten(start_dim=1).sum(1)
        
        # NOTE: we want to minimize this
        boundary_loss = -boundary_loss
        
        # Compute foreground likelhood
        fg_dist = Normal(y_nobg, self.fg_sigma)
        fg_likelihood = fg_dist.log_prob(x)
        
        kl = kl_z_what + kl_z_where + kl_z_pres + kl_z_depth
        
        if not self.arch.boundary_loss or globel_step > self.arch.bl_off_step:
            boundary_loss = boundary_loss * 0.0
        
        # For visualizating
        # Dimensionality check
        assert (
            (z_pres.size() == (B, self.arch.G**2, 1)) and
            (z_depth.size() == (B, self.arch.G**2, 1)) and
            (z_scale.size() == (B, self.arch.G**2, 2)) and
            (z_shift.size() == (B, self.arch.G**2, 2)) and
            (z_where.size() == (B, self.arch.G**2, 4)) and
            (z_what.size() == (B, self.arch.G**2, self.arch.z_what_dim))
        )
        log = {
            'fg': y_nobg,
            'glimpse':x_att.reshape(B, self.arch.G**2, *x_att.shape[-3:]).detach().cpu(),
            'z_what': z_what,
            'z_where': z_where,
            'z_pres': z_pres,
            'z_scale': z_scale,
            'z_shift': z_shift,
            'z_depth': z_depth,
            'z_pres_prob': torch.sigmoid(z_pres_logits),
            'prior_z_pres_prob': self.prior_z_pres_prob.unsqueeze(0),
            'o_att': o_att,
            'alpha_att_hat': alpha_att_hat,
            'alpha_att': alpha_att,
            'alpha_map': alpha_map,
            'boundary_loss': boundary_loss,
            'boundary_map': boundary_map,
            'importance_map_full_res_norm': importance_map_full_res_norm,
            
            'kl_z_what': kl_z_what,
            'kl_z_pres': kl_z_pres,
            'kl_z_scale': kl_z_scale,
            'kl_z_shift': kl_z_shift,
            'kl_z_depth': kl_z_depth,
            'kl_z_where': kl_z_where,
        }
        log.update(z_what_infos)
        log.update(sep_total_loss=sep_total_loss)
        return fg_likelihood, y_nobg, alpha_map, kl, boundary_loss, log


class ImgEncoderFg(nn.Module):
    """
    Foreground image encoder.
    """
    
    def __init__(self, arch):
        super(ImgEncoderFg, self).__init__()
        self.arch = arch
        assert self.arch.G in [4, 8, 16]
        # Adjust stride such that the output dimension of the volume matches (G, G, ...)
        last_stride = 2 if self.arch.G in [8, 4] else 1
        second_to_last_stride = 2 if self.arch.G in [4] else 1
        
        # Foreground Image Encoder in the paper
        # Encoder: (B, C, Himg, Wimg) -> (B, E, G, G)
        # G is H=W in the paper
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1) if arch.img_shape[0] == 128 else nn.Conv2d(3,16,3,1,1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 3, second_to_last_stride, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 256, 3, last_stride, 1),
            nn.CELU(),
            nn.GroupNorm(32, 256),
            nn.Conv2d(256, self.arch.img_enc_dim_fg, 1),
            nn.CELU(),
            nn.GroupNorm(16, self.arch.img_enc_dim_fg)
        )
        
        # Residual Connection in the paper
        # Remark: this residual connection is not important
        # Lateral connection (B, E, G, G) -> (B, E, G, G)
        self.enc_lat = nn.Sequential(
            nn.Conv2d(self.arch.img_enc_dim_fg, self.arch.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, self.arch.img_enc_dim_fg),
            nn.Conv2d(self.arch.img_enc_dim_fg, self.arch.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, self.arch.img_enc_dim_fg)
        )
        
        # Residual Encoder in the paper
        # Remark: also not important
        # enc + lateral -> enc (B, 2*E, G, G) -> (B, 128, G, G)
        self.enc_cat = nn.Sequential(
            nn.Conv2d(self.arch.img_enc_dim_fg * 2, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128)
        )
        
        # Image encoding -> latent distribution parameters (B, 128, G, G) -> (B, D, G, G)
        self.z_scale_net = nn.Conv2d(128, (self.arch.z_where_scale_dim) * 2, 1)
        self.z_shift_net = nn.Conv2d(128, (self.arch.z_where_shift_dim) * 2, 1)
        self.z_pres_net = nn.Conv2d(128, self.arch.z_pres_dim, 1)
        self.z_depth_net = nn.Conv2d(128, self.arch.z_depth_dim * 2, 1)
        
        # (G, G). Grid center offset. (offset_x[i, j], offset_y[i, j]) is the center for cell (i, j)
        offset_y, offset_x = torch.meshgrid([torch.arange(self.arch.G), torch.arange(self.arch.G)])
        
        # (2, G, G). I do this just to ensure that device is correct.
        self.register_buffer('offset', torch.stack((offset_x, offset_y), dim=0).float())
    
    def forward(self, x, tau):
        """
        Given image, infer z_pres, z_depth, z_where

        :param x: (B, 3, H, W)
        :param tau: temperature for the relaxed bernoulli
        :return
            z_pres: (B, G*G, 1)
            z_depth: (B, G*G, 1)
            z_scale: (B, G*G, 2)
            z_shift: (B, G*G, 2)
            z_where: (B, G*G, 4)
            z_pres_logits: (B, G*G, 1)
            z_depth_post: Normal, (B, G*G, 1)
            z_scale_post: Normal, (B, G*G, 2)
            z_shift_post: Normal, (B, G*G, 2)
        """
        B = x.size(0)
        
        # (B, C, H, W)
        img_enc = self.enc(x)
        # (B, E, G, G)
        lateral_enc = self.enc_lat(img_enc)
        # (B, 2E, G, G) -> (B, 128, H, W)
        cat_enc = self.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))
        
        def reshape(*args):
            """(B, D, G, G) -> (B, G*G, D)"""
            out = []
            for x in args:
                B, D, G, G = x.size()
                y = x.permute(0, 2, 3, 1).view(B, G * G, D)
                out.append(y)
            return out[0] if len(args) == 1 else out
        
        # Compute posteriors
        
        # (B, 1, G, G)
        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(cat_enc))
        # (B, 1, G, G) - > (B, G*G, 1)
        z_pres_logits = reshape(z_pres_logits)
        z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)
        # Unbounded
        z_pres_y = z_pres_post.rsample()
        # in (0, 1)
        z_pres = torch.sigmoid(z_pres_y)
        
        # (B, 1, G, G)
        z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
        # (B, 1, G, G) -> (B, G*G, 1)
        z_depth_mean, z_depth_std = reshape(z_depth_mean, z_depth_std)
        z_depth_std = F.softplus(z_depth_std)
        z_depth_post = Normal(z_depth_mean, z_depth_std)
        # (B, G*G, 1)
        z_depth = z_depth_post.rsample()
        
        # (B, 2, G, G)
        scale_std_bias = 1e-15
        z_scale_mean, _z_scale_std = self.z_scale_net(cat_enc).chunk(2, 1)
        z_scale_std = F.softplus(_z_scale_std) + scale_std_bias
        # (B, 2, G, G) -> (B, G*G, 2)
        z_scale_mean, z_scale_std = reshape(z_scale_mean, z_scale_std)
        z_scale_post = Normal(z_scale_mean, z_scale_std)
        z_scale = z_scale_post.rsample()
        
        # (B, 2, G, G)
        z_shift_mean, z_shift_std = self.z_shift_net(cat_enc).chunk(2, 1)
        z_shift_std = F.softplus(z_shift_std)
        # (B, 2, G, G) -> (B, G*G, 2)
        z_shift_mean, z_shift_std = reshape(z_shift_mean, z_shift_std)
        z_shift_post = Normal(z_shift_mean, z_shift_std)
        z_shift = z_shift_post.rsample()
        
        # scale: unbounded to (0, 1), (B, G*G, 2)
        z_scale = z_scale.sigmoid()
        # offset: (2, G, G) -> (G*G, 2)
        offset = self.offset.permute(1, 2, 0).view(self.arch.G ** 2, 2)
        # (B, G*G, 2) and (G*G, 2)
        # where: (-1, 1)(local) -> add center points -> (0, 2) -> (-1, 1)
        z_shift = (2.0 / self.arch.G) * (offset + 0.5 + z_shift.tanh()) - 1
        
        # (B, G*G, 4)
        z_where = torch.cat((z_scale, z_shift), dim=-1)
        
        # Check dimensions
        assert (
                (z_pres.size() == (B, self.arch.G ** 2, 1)) and
                (z_depth.size() == (B, self.arch.G ** 2, 1)) and
                (z_shift.size() == (B, self.arch.G ** 2, 2)) and
                (z_scale.size() == (B, self.arch.G ** 2, 2)) and
                (z_where.size() == (B, self.arch.G ** 2, 4))
        )
        
        return z_pres, z_depth, z_scale, z_shift, z_where, \
               z_pres_logits, z_depth_post, z_scale_post, z_shift_post

class ZWhatEnc(nn.Module):
    
    def __init__(self, arch):
        super(ZWhatEnc, self).__init__()
        self.arch = arch
        if arch.get('use_zwhat_mlp', False):
          assert arch.glimpse_size == 16
          enc_what_input_dim = 128
          self.enc_cnn = nn.Sequential(
            nn.Flatten(),
            create_mlp(3*16*16, enc_what_input_dim, [128, 128], return_seq=True, activation_fn=nn.CELU),
            nn.CELU())
        elif arch.glimpse_size == 8:
          self.enc_cnn = nn.Flatten() 
          enc_what_input_dim = 8*8*3
        elif arch.glimpse_size == 32:
          self.enc_cnn = nn.Sequential(
              nn.Conv2d(3, 16, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(4, 16),
              nn.Conv2d(16, 32, 4, 2, 1),
              nn.CELU(),
              nn.GroupNorm(8, 32),
              nn.Conv2d(32, 32, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(4, 32),
              nn.Conv2d(32, 64, 4, 2, 1),
              nn.CELU(),
              nn.GroupNorm(8, 64),
              nn.Conv2d(64, 128, 4, 2, 1),
              nn.CELU(),
              nn.GroupNorm(8, 128),
              nn.Conv2d(128, 256, 4),
              nn.CELU(),
              nn.GroupNorm(16, 256),
          )
          enc_what_input_dim = 256
        elif arch.glimpse_size == 16:
          self.enc_cnn = nn.Sequential(
              nn.Conv2d(3, 32, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(4, 32),
              nn.Conv2d(32, 64, 4, 2, 1),
              nn.CELU(),
              nn.GroupNorm(8, 64),
              nn.Conv2d(64, 128, 4, 2, 1),
              nn.CELU(),
              nn.GroupNorm(8, 128),
              nn.Conv2d(128, 256, 4),
              nn.CELU(),
              nn.GroupNorm(16, 256),
          )
          enc_what_input_dim = 256
        z_what_type = self.arch.get('z_what_type', 'VAE')
        if z_what_type == 'VAE':   
          enc_what_cls = VAE
        elif z_what_type == 'GMVAE':
          enc_what_cls = GMVAE
        elif z_what_type == 'SimpleGMVAE':
          enc_what_cls = SimpleGMVAE
        elif z_what_type == 'SimpleGMVAEV2':
          enc_what_cls = SimpleGMVAEV2
        print(f'Using ZWhatEncoder {enc_what_cls}')
        self.enc_what = enc_what_cls(enc_what_input_dim, self.arch.z_what_dim, self.arch)
    
    def forward(self, x, **kwargs):
        """
        Encode a (32, 32) glimpse into z_what

        :param x: (B, C, H, W)
        :return:
            z_what: (B, D)
            z_what_post: (B, D)
        """
        x = self.enc_cnn(x).flatten(start_dim=1)
        z_what, z_what_kl, infos = self.enc_what(x, **kwargs)
        return z_what, z_what_kl, infos
        
        # (B, D), (B, D)
        # z_what_mean, z_what_std = self.enc_what(x.flatten(start_dim=1)).chunk(2, -1)
        # z_what_std = F.softplus(z_what_std)
        # z_what_post = Normal(z_what_mean, z_what_std)
        # z_what = z_what_post.rsample()
        
        # return z_what, z_what_post


class GlimpseDec(nn.Module):
    """Decoder z_what into reconstructed objects"""
    
    def __init__(self, arch):
        super(GlimpseDec, self).__init__()
        self.arch = arch
        # I am using really deep network here. But this is overkill
        if arch.glimpse_size == 32:
          self.dec = nn.Sequential(
              nn.Conv2d(self.arch.z_what_dim, 256, 1),
              nn.CELU(),
              nn.GroupNorm(16, 256),
              
              nn.Conv2d(256, 128 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(16, 128),
              nn.Conv2d(128, 128, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(16, 128),
              
              nn.Conv2d(128, 128 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(16, 128),
              nn.Conv2d(128, 128, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(16, 128),
              
              nn.Conv2d(128, 64 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(8, 64),
              nn.Conv2d(64, 64, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(8, 64),
              
              nn.Conv2d(64, 32 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(8, 32),
              nn.Conv2d(32, 32, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(8, 32),
              
              nn.Conv2d(32, 16 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(4, 16),
              nn.Conv2d(16, 16, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(4, 16),
          )
          self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)
          self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)
        elif arch.glimpse_size == 16:
          self.dec = nn.Sequential(
              nn.Conv2d(self.arch.z_what_dim, 256, 1),
              nn.CELU(),
              nn.GroupNorm(16, 256),
              
              nn.Conv2d(256, 128 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(16, 128),
              nn.Conv2d(128, 128, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(16, 128),
              
              nn.Conv2d(128, 64 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(8, 64),
              nn.Conv2d(64, 64, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(8, 64),
              
              nn.Conv2d(64, 32 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(8, 32),
              nn.Conv2d(32, 32, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(8, 32),
              
              nn.Conv2d(32, 16 * 2 * 2, 1),
              nn.PixelShuffle(2),
              nn.CELU(),
              nn.GroupNorm(4, 16),
              nn.Conv2d(16, 16, 3, 1, 1),
              nn.CELU(),
              nn.GroupNorm(4, 16),
          )
          self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)
          self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)
        elif arch.glimpse_size == 8:
          self.dec = nn.Sequential(
              nn.Flatten(), 
              create_mlp(self.arch.z_what_dim, 256, [256, 256], return_seq=True),
              LastDimReshape(shape=(4,8,8)),
              nn.Sigmoid())
          self.dec_o = lambda x: x[:, :3] 
          self.dec_alpha = lambda x: x[:, 3:4]
    
    def forward(self, x):
        """
        Decoder z_what into glimpse

        :param x: (B, D)
        :return:
            o_att: (B, 3, H, W)
            alpha_att: (B, 1, H, W)
        """
        x = self.dec(x.view(x.size(0), -1, 1, 1))
        
        o = torch.sigmoid(self.dec_o(x))
        alpha = torch.sigmoid(self.dec_alpha(x))
        
        return o, alpha

