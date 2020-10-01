import os
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset, pil_loader
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms

animal_label = {
    "bird":         0,
    "butterfly":    1,
    "cat":          2,
    "crocodile":    3,
    "cow":          4,
    "dog":          5,
    "dolphine":     6,
    "duck":         7,
    "elephant":     8,
    "fish":         9,
    "hen":          10,
    "leopard":      11,
    "monkey":       12,
    "rabbit":       13,
    "rat":          14,
    "spider":       15,
    "tortoise":     16
}


class AnimalDataset(VisionDataset):
    """
    Animal Silhouette dataset
    """

    def __init__(self, root_dir, extensions, transforms=None, target_transforms=None, test=False):
        super(AnimalDataset, self).__init__(root_dir,
                                            transforms,
                                            target_transforms)
        self.root_dir = root_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.extensions = extensions
        classes, class_to_idx = self._find_classes(self.root_dir)
        samples = make_dataset(root_dir, class_to_idx, self.extensions, is_valid_file=None)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        #####1D conv#############
        # sample = sample.flatten()
        # print(sample.dtype)
        sample = np.transpose(sample)
        sample = sample.copy()
        sample = torch.tensor(sample)
        target = torch.tensor(target)
        return sample, target

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        class_to_idx = animal_label
        return classes, class_to_idx


# data_dir = "animal_silhouette_training"
# ext = "tif"
# transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# dataset = AnimalDataset(data_dir, pil_loader, ext, transform)
# for i in range(1):
#     data = dataset.__getitem__(i)
#     print(data[0], data[1])
