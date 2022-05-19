import torch
"""
ppo_params:
  _norm_adv, _dual_clip, _eps_clip, _value_clip, _weight_vf, _weight_ent
"""
def ppo_loss(dist, value, adv, act, logp_old, v_s, v_targ, ppo_params):
    assert dist is not None or value is not None
    if dist is not None:
      if ppo_params._norm_adv:
          mean, std = adv.mean(), adv.std()
          adv = (adv - mean) / std  # per-batch norm
      ratio = (dist.log_prob(act) - logp_old).exp().float()
      ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
      with torch.no_grad():
        clip_fraction = torch.mean((torch.abs(ratio - 1) > ppo_params._eps_clip).float())
        log_ratio = dist.log_prob(act) - logp_old
        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
      surr1 = ratio * adv
      surr2 = ratio.clamp(1.0 - ppo_params._eps_clip, 1.0 + ppo_params._eps_clip) * adv
      if ppo_params._dual_clip:
          clip1 = torch.min(surr1, surr2)
          clip2 = torch.max(clip1, ppo_params._dual_clip * adv)
          clip_loss = -torch.where(adv < 0, clip2, clip1).mean()
      else:
          clip_loss = -torch.min(surr1, surr2).mean()
      ent_loss = dist.entropy().mean()
    else:
      clip_loss = torch.Tensor([0]).to(adv.device).sum()
      ent_loss  = clip_loss
      approx_kl_div, clip_fraction = clip_loss, clip_loss
    # calculate loss for critic
    if value is not None:
      value = value.flatten()
      if ppo_params._value_clip:
          v_clip = v_s + (value -
                            v_s).clamp(-ppo_params._eps_clip, ppo_params._eps_clip)
          vf1 = (v_targ - value).pow(2)
          vf2 = (v_targ - v_clip).pow(2)
          vf_loss = torch.max(vf1, vf2).mean()
      else:
          vf_loss = (v_targ - value).pow(2).mean()
    else:
      vf_loss = torch.Tensor([0]).to(adv.device).sum()
    # calculate regularization and overall loss
    loss = clip_loss + ppo_params._weight_vf * vf_loss \
        - ppo_params._weight_ent * ent_loss
    return loss, {
        "approx_kl_div": approx_kl_div.item(),
        "clip_fraction": clip_fraction.item(),
        "loss": loss.item(),
        "loss/clip": clip_loss.item(),
        "loss/vf": vf_loss.item(),
        "loss/ent": ent_loss.item(),
    }

