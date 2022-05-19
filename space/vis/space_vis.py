from torch.utils.tensorboard import SummaryWriter
import imageio
import pickle
import joblib
import numpy as np
import torch
from space.utils import spatial_transform
from space.vis.utils import boxes, bbox_in_one
from attrdict import AttrDict
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import matplotlib.image as mpimg
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
class SpaceVis:
    def __init__(self):
      # objects = ['sand', 'zombie', 'player', 'cow', 'stone']
      objects = ['sand', 'cow', 'player', 'tree', 'stone']
      def read_img(k):
        img = mpimg.imread(os.path.join('/lustre/S/yiqi/work/ILP/object/obj_rn2/crafter/assets', f'{k}.png'))
        img = np.asarray(img)
        img = torch.as_tensor(img).permute(2,0,1)
        return img
      def post_process(img, bg):
        if img.shape[0] == 4:
          alpha = img[-1:]
          img = bg * (1-alpha) + alpha * img[:3]
        return img
      assets = {k: read_img(k) for k in objects}
      print({k:v.shape for k,v in assets.items()})
      def select_bg(k):
        if k in ['coal','iron','diamond']:
          bg = assets['stone']
        else:
          if 'grass' in assets:
            bg = assets['grass']
          else:
            bg = assets['sand']
        return bg
      assets = {k: post_process(v, select_bg(k)) for k,v in assets.items()}
      self.all_glimps = torch.stack([assets[k] for k in objects], dim=0)

    @torch.no_grad()
    def cat_vis(self, model, writer, global_step, device):
      if model.fg_module.z_cat_net:
        z_what_net = model.fg_module.z_cat_net
      else:
        z_what_net = model.fg_module.z_what_net
      *args, z_what_infos = z_what_net(self.all_glimps.to(device))
      if z_what_infos.get('enc_x_cat', None):
        z_what_cat = z_what_infos['enc_x_cat'].permute(1,0)
        writer.add_image(f'CatInfo', z_what_cat.unsqueeze(0), global_step)

    @torch.no_grad()
    def train_vis(self, model, writer: SummaryWriter, log, global_step, mode, num_batch=10, log_scalor=True):
        """
        """
        B = num_batch
        
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
                if isinstance(log[key], torch.Tensor) and log[key].ndim > 0:
                    log[key] = log[key][:num_batch]
        log = AttrDict(log)
        
        if model.fg_module.z_cat_net:
          print('Z cat plotted.')
          glimpse = log.glimpse.to(next(model.parameters()).device)
          _, _, infos = model.fg_module.z_cat_net(glimpse.reshape(-1, *glimpse.shape[-3:]), recon=False)
          z_cat = infos['enc_x_cat'].argmax(-1).view(-1)
          box = boxes[z_cat]
          fg_box = bbox_in_one(
              log.fg, log.z_pres, log.z_scale, log.z_shift, box
          )
        else:
          # (B, 3, H, W)
          fg_box = bbox_in_one(
              log.fg, log.z_pres, log.z_scale, log.z_shift
          )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, bg, recon, fg, fg_box, alpha_map, masked_comps, masks, comps], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.view(B*N, 3, H, W)
        
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/#0-separations', grid_image, global_step)
        
        grid_image = make_grid(log.imgs, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/1-image', grid_image, global_step)
        
        grid_image = make_grid(log.y, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/2-reconstruction_overall', grid_image, global_step)
        
        grid_image = make_grid(log.bg, 5, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/3-background', grid_image, global_step)

        if not log_scalor:
          return
        
        mse = (log.y - log.imgs) ** 2
        mse = mse.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth, kl_bg = (
            log['log_like'].mean(), log['kl_z_what'].mean(), log['kl_z_where'].mean(),
            log['kl_z_pres'].mean(), log['kl_z_depth'].mean(), log['kl_bg'].mean()
        )
        loss_boundary = log.boundary_loss.mean()
        loss = log.loss.mean()
        
        count = log.z_pres.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        writer.add_scalar(f'{mode}/mse', mse.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/loss', loss, global_step=global_step)
        writer.add_scalar(f'{mode}/count', count, global_step=global_step)
        writer.add_scalar(f'{mode}/log_like', log_like.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/loss_boundary', loss_boundary.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/What_KL', kl_z_what.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Where_KL', kl_z_where.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Pres_KL', kl_z_pres.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Depth_KL', kl_z_depth.item(), global_step=global_step)
        writer.add_scalar(f'{mode}/Bg_KL', kl_bg.item(), global_step=global_step)
        for loss_name in log:
          if loss_name.startswith('gmvae'):
            writer.add_scalar(f'{mode}/{loss_name}', float(log[loss_name]), global_step=global_step)
            
    
    @torch.no_grad()
    def show_vis(self, model, dataset, indices, path, device):
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
        data = next(iter(dataloader))
        data = data.to(device)
        loss, log = model(data, 100000000)
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
        log = AttrDict(log)
        # (B, 3, H, W)
        fg_box = bbox_in_one(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, bg, recon,  fg, fg_box, alpha_map, masked_comps, masks, comps,], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.view(B*N, 3, H, W)
        
        # (3, H, W)
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        
        # (H, W, 3)
        image = torch.clamp(grid_image, 0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        imageio.imwrite(path, image)

    @torch.no_grad()
    def embedding_vis(self, embedding_dir, model, dataset, indices, device, valid_thres=0.8):
        writer = SummaryWriter(log_dir=embedding_dir, flush_secs=10)
        total_z_what, total_z_shift = [], []
        total_glimpse = []
        enc_cat = []

        for i in range(0, len(indices), 128):
          print(f'Cal {i} to {i+128}...')
          sub_indices = indices[i:min(i+128, len(indices))]
          sub_dataset = Subset(dataset, sub_indices)
          dataloader = DataLoader(sub_dataset, batch_size=len(sub_indices), shuffle=False)
          data = next(iter(dataloader))
          data = data.to(device)
          loss, log = model(data, 100000000)
          for key, value in log.items():
              if isinstance(value, torch.Tensor):
                  log[key] = value.detach().cpu()
          log = AttrDict(log)
          log['loss'] = 0

          z_what, z_pres, z_shift, glimpse = log.z_what, log.z_pres, log.z_shift, log.glimpse
          z_pres = z_pres.flatten()
          z_what = z_what.reshape(z_pres.shape[0], -1)
          z_shift = z_shift.reshape(-1, 2)

          valid_mask = z_pres >= valid_thres
          valid_z_what = z_what[valid_mask]
          valid_z_shift = z_shift[valid_mask]
          valid_glimpse = glimpse[valid_mask]
          if 'enc_x_cat' in log:
            valid_enc_cat = log['enc_x_cat'][valid_mask]
            enc_cat.append(valid_enc_cat)
          total_z_shift.append(valid_z_shift.detach().cpu())
          total_z_what.append(valid_z_what.detach().cpu())
          total_glimpse.append(valid_glimpse)
        valid_z_what = torch.cat(total_z_what, dim=0)
        valid_z_shift = torch.cat(total_z_shift, dim=0)
        valid_glimpse = torch.cat(total_glimpse, dim=0)
        print(f'ZShift ({valid_z_shift.shape}):')
        if len(enc_cat) > 0:
          print('Using Encoder Cat')
          enc_cat = torch.cat(enc_cat, dim=0)
          labels = enc_cat.argmax(-1).numpy()
          print(enc_cat[0:8])
          print(enc_cat.mean(0))
        else:
          kmeans =  KMeans(init="k-means++", n_clusters=8, max_iter=1000)
          kmeans =  kmeans.fit(valid_z_what)
          joblib.dump(kmeans, os.path.join(embedding_dir, 'kmeans.model'))
          print('K means calculated.')
          labels = kmeans.predict(valid_z_what)
        torch.save(dict(z_what=valid_z_what.detach().cpu(),
                        z_shift=valid_z_shift.detach().cpu(), 
                        labels=torch.as_tensor(labels),
                        glimpse=valid_glimpse.detach().cpu()),
                  os.path.join(embedding_dir, 'embeddings.pkl'))
        writer.add_embedding(valid_z_what, metadata=labels, label_img=valid_glimpse)
        # self.train_vis(writer, log, 10000000, 'embedding', 10, log_scalor=False)







         



