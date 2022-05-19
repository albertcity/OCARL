import pickle
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
import warnings
from tianshou.data import Batch, VectorReplayBuffer, ReplayBuffer, to_numpy, to_torch, to_torch_as

class CustomDataset(Dataset):
    def __init__(self, path, start=0, end=-1, image_size=(128, 128), transform=None, to_tensor=True, gamelist=None):
        print(path, image_size)
        self.path = path
        try:
          self.data = VectorReplayBuffer.load_hdf5(path).sample(0)[0].obs[start:end]
        except Exception as e:
          print(f'VecBuffer Load failed with path={path}')
          print(e)
          self.data = ReplayBuffer.load_hdf5(path).sample(0)[0].obs[start:end]
        self.image_size = tuple(image_size)
        self.transform = transform  # data augmentation
        self.to_tensor = to_tensor

        print(f'Dataset Path: {path}')
        print('++++++++++++++++++++++++++++ Procgen Data Loaded ++++++++++++++++++++++++++++++++++++')

    def __getitem__(self, index):
        image = self.data[index]
        assert image.dtype == np.uint8
        obs = image
        # resize if necessary
        if tuple(image.shape[:2]) != self.image_size:
            warnings.warn('Resize image from {} to {}'.format(image.shape, self.image_size))
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            image_pil = self.transform(image)
            image = np.asarray(image_pil)

        assert image.dtype == np.uint8
        image = np.asarray(image, dtype=np.float32) / 255.

        if self.to_tensor:
            image = np.transpose(image, [2, 0, 1])  # (c, h, w)
            image = torch.tensor(image, dtype=torch.float32)

        return image

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return '{:s}: {:d} images.'.format(self.__class__.__name__, len(self))
