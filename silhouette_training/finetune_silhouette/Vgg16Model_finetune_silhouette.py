import torch
import torchvision
import torchvision.transforms as transforms
from project2.animal_dataset import AnimalDataset
from torchvision.datasets.folder import pil_loader
from project3.raster_points import RasterPointsTransformation
import matplotlib.pyplot as plt
import numpy as np


class ConfigFinetuneImagenetSilhouette:
    def __init__(self, dataset_dir, shuffle):
        # 1.string parameters
        self.__train_dir = dataset_dir
        self.model_name = "vgg16_bn"

        # 2.numeric parameters
        self.epochs = 200
        self.start_epoch = 0
        self.batch_size = 16
        self.momentum = 0.9
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.interval = 10
        self.workers = 2
        self.num_classes = 1000  # output dimension

        # 3.boolean parameters
        # evaluate = False
        self.pretrained = True
        # resume = False

        # transform - imagenet intact images
        self.__transform = transforms.Compose([transforms.Resize(224),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # transform - low resolution
        self.__transform_low_resolution = transforms.Compose([RasterPointsTransformation(32),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.485, 0.456, 0.406),
                                                                                   (0.229, 0.224, 0.225))])
        # imagenet dataloader

        self.finetune_animal_dataset = AnimalDataset(root_dir=self.__train_dir,
                                                     # loader=pil_loader, (don't apply if low_resolution)
                                                     extensions="tif",
                                                     transforms=self.__transform_low_resolution)

        self.loader_imagenet = torch.utils.data.DataLoader(self.finetune_animal_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=shuffle,
                                                           num_workers=self.workers)

#
# config = ConfigFinetuneImagenetSilhouette()
# dataset = config.finetune_animal_dataset
# iterator = iter(dataset)
# data, label = next(iterator)
# data = np.array(data)
# data = np.transpose(data, (1, 2, 0))
# plt.imshow(data)
# plt.show()
