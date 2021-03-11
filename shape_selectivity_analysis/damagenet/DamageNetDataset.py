import os
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset, pil_loader
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
torch.manual_seed(os.getenv("SEED"))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DamageNetDataset(VisionDataset):
    """
    DamageNet dataset
    """

    def __init__(self, root_dir, label_dir, transforms=None):
        super(DamageNetDataset, self).__init__(root_dir)
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transforms = transforms
        file_to_idx = self._find_classes(self.label_dir)
        samples = self.make_dataset(root_dir, file_to_idx)
        self.file_to_idx = file_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transforms is not None:
            sample = self.transforms(sample)

        #####1D conv#############
        # sample = sample.flatten()
        # print(sample.dtype)
        # sample = np.transpose(sample)
        # sample = sample.copy()
        # sample = torch.tensor(sample)
        # target = torch.tensor(target)
        return sample, target

    def _find_classes(self, dir):
        file_to_idx = {}
        with open(dir, "r") as f:
            for line in f.readlines():
                file_name, idx = line.split()
                file_to_idx[file_name] = int(idx)
        return file_to_idx

    def make_dataset(self, root_dir, file_to_idx):
        samples = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                ele = (root_dir + "\\" + file, file_to_idx[file])
                samples.append(ele)
        return samples


if __name__ == "__main__":
    dir = "D:\\projects\\summerProject2020\\project1\\DAmageNet\\DAmageNet"
    label = "D:\\projects\\summerProject2020\\project1\\DAmageNet\\val_damagenet.txt"
    dataset = DamageNetDataset(dir, label)
    print(dataset.samples)
