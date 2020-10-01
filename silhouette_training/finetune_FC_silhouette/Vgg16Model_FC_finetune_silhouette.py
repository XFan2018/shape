import torch
import torchvision
import torchvision.transforms as transforms
from project2.animal_dataset import AnimalDataset
from torchvision.datasets.folder import pil_loader


class ConfigFinetuneSilhouette:
    def __init__(self):
        # 1.string parameters
        self.__train_dir = "/Users/leo/PycharmProjects/summerProject/project2/animal_silhouette_training"
        self.model_name = "vgg16_bn"

        # 2.numeric parameters
        self.epochs = 1
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
        self.__transform = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # imagenet dataloader

        self.__finetune_animal_dataset = AnimalDataset(root_dir=self.__train_dir,
                                                       loader=pil_loader,
                                                       extensions="tif",
                                                       transforms=self.__transform)

        self.train_loader_imagenet = torch.utils.data.DataLoader(self.__finetune_animal_dataset,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=False,
                                                                 num_workers=self.workers)
