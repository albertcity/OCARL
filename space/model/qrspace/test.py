import torch
from torch.distributions import Normal, kl_divergence, Categorical
import numpy as np
import einops
from einops.layers.torch import Rearrange
import torch.nn as nn

K=8
cat_a = Categorical(probs=torch.as_tensor([1./K] * K))
cat_b = Categorical(probs=torch.as_tensor([1.] * 4 + [0] * (K-4)))

kl = kl_divergence(cat_b, cat_a)
print(kl)
print(-cat_b.entropy() - np.log(1/8))

# z_what_cat = torch.randn(100, 10)
# tau = 2.5
# z_what_cat_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(temperature=tau, logits=z_what_cat)
# z_what_cat_onehot = z_what_cat_dist.rsample()
# 
# z_what_cat_prior = torch.as_tensor([1./10] * 10) 
# z_what_cat_prior_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(probs=z_what_cat_prior, temperature=0.1)
# kl_z_what_cat = kl_divergence(z_what_cat_dist, z_what_cat_prior_dist)
# print(kl_z_what_cat)
