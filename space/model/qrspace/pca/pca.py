import torch
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np
from scipy import sparse
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os

def pred_cat(x, pca, kmeans):
  z = pca.transform(x.flatten(1))
  cat = kmeans.predict(z)
  return cat

N = 64
M = 2048
C = 4

colors = [[255,0,0], [255,255,0], [255,153,18],[255,127,80],[255,192,203],[255,0,255], [0,255,0], [0,255,255], [8,46,84], [0,199,140], [0,0,255], [160,32,240],[218,112,214]] 
colors = torch.Tensor(colors) / 255.
logdir = './log'
data = torch.load('/lustre/S/yiqi/work/ILP/object/obj_rn2/datasets/all_glimpse.pt')
data = data[:10000]
X = data.flatten(1)
transformer = IncrementalPCA(n_components=4, batch_size=256)
X_sparse = sparse.csr_matrix(X)
X_transformed = transformer.fit_transform(X_sparse)
kmeans =  KMeans(init="k-means++", n_clusters=C, max_iter=1000)
kmeans =  kmeans.fit(X_transformed)

joblib.dump(dict(pca=transformer, kmeans=kmeans), 'cat_pred.model')
labels = kmeans.predict(X_transformed[:M])

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir='./log', flush_secs=10)

writer.add_embedding(X_transformed[:M], metadata=labels, label_img=data[:M], global_step=0)

imgs = data[np.random.choice(len(data), N)]
cat = pred_cat(imgs, transformer, kmeans)
imgs[:,:,:3] = colors[cat].reshape(-1,3,1,1)
grid = make_grid(imgs, 8) 
writer.add_image('CatInfo', grid, 0)

for i in range(C):
  cat_imgs = data[:M][labels == i]
  cat_imgs = cat_imgs[:min(64, len(cat_imgs))]
  grid = make_grid(cat_imgs, 8)
  writer.add_image(f'CatImg/Cat{i}', grid, global_step=0)






         



