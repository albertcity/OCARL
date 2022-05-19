from torch.nn import functional as F
import matplotlib
matplotlib.ust('AGG')
import matplotlib.pyplot as plt
import gym
import procgen

# (H, W, 3)
img = gym.make('procgen:procgen-heist-v0').reset()
img_torch = torch.as_tensor(img_torch)
theta = torch.tensor([
    [1,0,-0.2],
    [0,1,-0.4]
], dtype=torch.float)
grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
output = F.grid_sample(img_torch.unsqueeze(0), grid)
new_img_torch = output[0]
plt.imshow(new_img_torch.numpy().transpose(1,2,0))
plt.show()
