import os
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset, pil_loader
from PIL import Image
from image_to_polygon import PolygonTransform
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class HemeraDataset(VisionDataset):
    """
    Animal Silhouette dataset
    """

    def __init__(self, root_dir, extensions, transforms=None, target_transforms=None):
        super(HemeraDataset, self).__init__(root_dir,
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
        class_to_idx = {"v1": 1, "v2": 2, "v3": 3}
        return classes, class_to_idx


if __name__ == "__main__":
    data_dir = "D:\projects\shape_dataset\Hemera"
    ext = "png"
    transform = transforms.Compose([PolygonTransform(int(32))])
    dataset = HemeraDataset(data_dir, ext, transform)
    dataset2 = HemeraDataset(data_dir, ext)
    data = dataset.__getitem__(25)
    data2 = dataset2.__getitem__(25)
    img = Image.fromarray(data2[0].detach().numpy().transpose())
    img.show()
    print(data[0], data[1])
    plt.scatter(data[0][0], data[0][1])
    plt.show()