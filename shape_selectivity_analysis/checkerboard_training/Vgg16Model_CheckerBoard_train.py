import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scrambleTransform import ScrambleTransform, ScrambleRandomTransform


class ConfigTrainImagenet:
    def __init__(self, dataset_path, epochs=None, lr=1e-3, shuffle=False, block_size=None):
        # 1.string parameters
        self.__dataset_dir = dataset_path
        self.model_name = "vgg16_bn"

        # 2.numeric parameters
        self.epochs = epochs
        self.start_epoch = 0
        self.batch_size = 16
        self.momentum = 0.9
        self.lr = lr
        self.weight_decay = 1e-4
        self.interval = 10
        self.workers = 2
        self.num_classes = 1000  # output dimension

        # 3.boolean parameters
        # evaluate = False
        self.pretrained = True
        self.shuffle = shuffle
        # resume = False

        # checkerboard transform
        self.checkerboard_transform = transforms.Compose([transforms.Resize(256),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor()])

        # checkerboard dataloader
        self.dataset = torchvision.datasets.ImageFolder(root=self.__dataset_dir,
                                                        transform=self.checkerboard_transform)

        self.loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=self.shuffle,
                                                  num_workers=self.workers)

        self.scrambled_transform = transforms.Compose([transforms.Resize(256),
                                                       transforms.CenterCrop(224),
                                                       ScrambleTransform(block_size),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                                            (0.229, 0.224, 0.225))])

        # scrambled dataloader
        self.scrambled_dataset = torchvision.datasets.ImageFolder(root=self.__dataset_dir,
                                                                  transform=self.scrambled_transform)

        self.scrambled_loader = torch.utils.data.DataLoader(self.scrambled_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=self.shuffle,
                                                            num_workers=self.workers)


if __name__ == "__main__":
    config = ConfigTrainImagenet("imagenet_val_testing_dataset")
    it = iter(config.loader)
    print(next(it))
