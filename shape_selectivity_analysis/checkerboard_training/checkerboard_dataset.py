import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset


class CheckerboardDataset(VisionDataset):
    """
    checkerboard dataset
    """

    def __init__(self, root_dir, extensions, transforms=None, target_transforms=None, test=False):
        super(CheckerboardDataset, self).__init__(root_dir,
                                                  transforms,
                                                  target_transforms)

        file_name_list = os.listdir(root_dir)
        labels_list = range(1000)
        self.dataset_labels = dict(zip(file_name_list, labels_list))
        self.root_dir = root_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.extensions = extensions
        classes, class_to_idx = self._find_classes(self.root_dir)
        samples = make_dataset(root_dir, class_to_idx, self.extensions, is_valid_file=None)
        print(samples)
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
        class_to_idx = self.dataset_labels
        return classes, class_to_idx


data_dir = "imagenet_val_testing_dataset"
ext = "JPEG"
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dataset = CheckerboardDataset(data_dir, ext, transform)

# for i in range(1):
#     data = dataset.__getitem__(i)
#     print(data[0], data[1])
