import os
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset, pil_loader
from PIL import Image
from torchvision import transforms

animal_label = {
    "bird":         17,
    "butterfly":    326,
    "cat":          281,
    "crocodile":    49,
    "cow":          345,
    "dog":          207,
    "dolphine":     148,
    "duck":         97,
    "elephant":     385,
    "fish":         0,
    "hen":          8,
    "leopard":      288,
    "monkey":       373,
    "rabbit":       331,
    "rat":          333,
    "spider":       74,
    "tortoise":     37
}


class AnimalDataset(VisionDataset):
    """
    Animal Silhouette dataset
    """

    def __init__(self, root_dir, extensions, transforms=None, target_transforms=None, loader=None):
        super(AnimalDataset, self).__init__(root_dir,
                                            transforms,
                                            target_transforms)
        self.root_dir = root_dir
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.extensions = extensions
        self.loader = loader
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
        if self.loader is None:
            sample = Image.open(path)
        else:
            sample = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return sample, target

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        class_to_idx = animal_label
        return classes, class_to_idx


data_dir = "animal_silhouette_training"
ext = "tif"
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = AnimalDataset(data_dir, ext, transform)
# for i in range(1):
#     data = dataset.__getitem__(i)
#     print(data[0], data[1])
print(dataset.samples)
