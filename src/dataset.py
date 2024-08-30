import os
import PIL

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_dir: str, label_filepath: str = None, transform=None):

        # Load labels is specified ('train' mode), else set mode to 'test'
        if label_filepath is not None:
            with open(label_filepath, 'r') as label_file:
                labels = label_file.readlines()
            labels = [float(line.strip()) for line in labels]
            mode = "train"
        else:
            labels = None
            mode = "test"

        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.labels = labels
        self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        # Load image
        img = PIL.Image.open(self.img_paths[idx])
        img = torch.from_numpy(np.array(img, copy=True).transpose((2, 0, 1))).to(torch.float)
        # Get corresponding label
        label = self.labels[idx] if self.mode == "train" else torch.Tensor([])
        # Apply transform if necessary
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_paths)
    