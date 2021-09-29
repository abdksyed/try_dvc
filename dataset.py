
import torch
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Create Cat and Dog Classification Dataser
class CatDog(ImageFolder):
    def __init__(self, root, transform=None, train=True):
        """
        Args:
            root: the root directory of the images
            transform: the transform to be applied to the images
            train: if True, the dataset will be used for training
        """
        self.root = root
        self.train = train
        self.transform = transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.train_data, self.train_labels = self.processed_folder(os.path.join(self.root, 'train'))
        self.validation_data, self.validation_labels = self.processed_folder(os.path.join(self.root, 'validation'))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.validation_data[index], self.validation_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.validation_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'train')) and os.path.exists(os.path.join(self.root, 'validation'))


a = CatDog(root='F:/#Coding/try_dvc/data', train=True)
print(a)