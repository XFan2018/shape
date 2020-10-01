import torch
import torchvision
import torchvision.transforms as transforms
from project1.scrambleTransform import ScrambleTransform, ScrambleRandomTransform, HorizontalScrambleTransform
import PIL.Image as Image


class ConfigTestImagenet:
    def __init__(self, block_size=224):
        # 1.string parameters
        self.__val_dir = "D:\\projects\\summerProject2020\\project1\\imagenet_val_validation_dataset"
        self.model_name = "vgg16_bn"
        # weights = "./checkpoints/"
        # best_models = weights + "best_model/"

        # 2.numeric parameters
        self.batch_size = 16
        self.interval = 10
        self.workers = 2
        self.num_classes = 1000  # output dimension
        self.block_size = block_size

        # 3.boolean parameters
        # evaluate = False
        self.pretrained = True
        # resume = False

        # transform1 - imagenet scrambled images
        self.__transform1 = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                ScrambleTransform(block_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # transform2 - imagenet images
        self.__transform2 = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # transform3 - imagenet horizontal scrambled images
        self.__transform3 = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                HorizontalScrambleTransform(block_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # imagenet dataloader

        self.__test_dataset_imagenet = torchvision.datasets.ImageFolder(root=self.__val_dir,
                                                                        transform=self.__transform2)

        self.__test_dataset_imagenet_scrambled = torchvision.datasets.ImageFolder(root=self.__val_dir,
                                                                                  transform=self.__transform1)

        self.__test_dataset_imagenet_horizontal_scrambled = torchvision.datasets.ImageFolder(root=self.__val_dir,
                                                                                             transform=self.__transform3)

        self.test_loader_imagenet = torch.utils.data.DataLoader(self.__test_dataset_imagenet,
                                                                batch_size=self.batch_size,
                                                                shuffle=False,
                                                                num_workers=self.workers)

        self.test_loader_imagenet_scrambled = torch.utils.data.DataLoader(self.__test_dataset_imagenet_scrambled,
                                                                          batch_size=self.batch_size,
                                                                          shuffle=False,
                                                                          num_workers=self.workers)

        self.test_loader_imagenet_horizontal_scrambled = torch.utils.data.DataLoader(
            self.__test_dataset_imagenet_horizontal_scrambled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers)
