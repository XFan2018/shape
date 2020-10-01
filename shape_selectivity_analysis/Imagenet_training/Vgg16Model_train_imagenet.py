import torch
import torchvision
import torchvision.transforms as transforms
from project1.scrambleTransform import ScrambleTransform, ScrambleRandomTransform


class ConfigTrainImagenet:
    def __init__(self, data_path, epochs, learning_rate):
        # 1.string parameters
        self.__train_dir = data_path
        # 2.numeric parameters
        self.epochs = epochs
        self.batch_size = 16
        self.momentum = 0.9
        self.lr = learning_rate
        self.weight_decay = 1e-5
        self.interval = 10
        self.workers = 2
        self.num_classes = 1000  # output dimension
        # transform1 - imagenet scrambled images
        self.__transform1 = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                ScrambleRandomTransform(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # transform2 - imagenet intact images
        self.__transform2 = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # imagenet dataloader

        self.__train_dataset_imagenet_scrambled = torchvision.datasets.ImageFolder(root=self.__train_dir,
                                                                                   transform=self.__transform1)

        self.__train_dataset_imagenet = torchvision.datasets.ImageFolder(root=self.__train_dir,
                                                                         transform=self.__transform2)

        self.train_loader_imagenet_scrambled = torch.utils.data.DataLoader(self.__train_dataset_imagenet_scrambled,
                                                                           batch_size=self.batch_size,
                                                                           shuffle=True,
                                                                           num_workers=self.workers)

        self.train_loader_imagenet = torch.utils.data.DataLoader(self.__train_dataset_imagenet,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 num_workers=self.workers)
