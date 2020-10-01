import torch
import torchvision
import torchvision.transforms as transforms
from project1.DamageNetDataset import DamageNetDataset
from project1.scrambleTransform import ScrambleTransform, ScrambleRandomTransform, HorizontalScrambleTransform
import PIL.Image as Image


class ConfigTestImagenet:
    def __init__(self, block_size=224):
        # 1.string parameters
        self._dir = "D:\\projects\\summerProject2020\\project1\\DAmageNet\\DAmageNet"
        self._label = "D:\\projects\\summerProject2020\\project1\\DAmageNet\\val_damagenet.txt"
        self.model_name = "vgg16_bn"

        # 2.numeric parameters
        self.batch_size = 16
        self.interval = 10
        self.workers = 2
        self.num_classes = 1000  # output dimension
        self.block_size = block_size

        # 3.boolean parameters
        self.pretrained = True

        # transform1 - imagenet scrambled images
        self._transform1 = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self._test_dataset = DamageNetDataset(self._dir, self._label, self._transform1)

        self.test_loader = torch.utils.data.DataLoader(self._test_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.workers)
