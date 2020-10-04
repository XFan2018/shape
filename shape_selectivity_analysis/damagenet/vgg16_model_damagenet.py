import torch
import torchvision
import torchvision.transforms as transforms
from DamageNetDataset import DamageNetDataset


class ConfigTestImagenet:
    def __init__(self, damagenet_dataset, damagenet_label, batch_size, workers):
        # 1.string parameters
        self._dir = damagenet_dataset
        self._label = damagenet_label

        # 2.numeric parameters
        self.batch_size = batch_size
        self.workers = workers

        # transform1 - imagenet scrambled images
        self._transform = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self._test_dataset = DamageNetDataset(self._dir, self._label, self._transform)

        self.test_loader = torch.utils.data.DataLoader(self._test_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.workers)
