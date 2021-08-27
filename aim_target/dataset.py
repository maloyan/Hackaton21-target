import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TargetDataset(Dataset):
    def __init__(self, data_path, target, is_test=False, augmentation=None, classes=7):
        super().__init__()
        self.data_path = data_path
        self.target = target
        self.is_test = is_test
        self.augmentation = augmentation
        self.classes = classes

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        image = cv2.imread(self.data_path[item])

        if self.is_test:
            return torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float)

        target = [0] * self.classes
        target[self.target[item]] = 1

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]
        return torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float), torch.tensor(
            target, dtype=torch.float
        )
