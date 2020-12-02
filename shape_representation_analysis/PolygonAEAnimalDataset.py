import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from image_to_polygon import RandomRotatePoints, RandomFlipPoints, IndexRotate


class PolygonAEAnimalDataset():
    """
    Silhouette Polygon dataset
    """

    def __init__(self, root_dir, root_dir_label, transforms=None, twoDim=False):
        self.root_dir = root_dir
        self.transforms = transforms
        samples = pd.read_csv(root_dir, header=None).values
        labels = pd.read_csv(root_dir_label, header=None).values
        self.labels = labels
        self.samples = samples
        self.twoDim = twoDim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]

        if self.twoDim:
            sample = sample.reshape(2, 32)

        if self.transforms is not None:
            sample = self.transforms(sample)

        sample = sample.copy()
        label = int(label.copy())
        sample = torch.tensor(sample)
        label = torch.tensor(label).long()

        return sample, label


class TurningAngleAEAnimalDataset():
    """
    Silhouette Turning angle dataset
    """

    def __init__(self, root_dir, root_dir_label, transforms=None, twoDim=False):
        self.root_dir = root_dir
        self.transforms = transforms
        samples = pd.read_csv(root_dir, header=None).values
        labels = pd.read_csv(root_dir_label, header=None).values
        self.labels = labels
        self.samples = samples
        self.twoDim = twoDim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.twoDim:
            sample = sample.reshape(1, 32)
        sample = sample.copy()
        label = int(label.copy())
        sample = torch.tensor(sample)
        label = torch.tensor(label).long()

        return sample, label


if __name__ == "__main__":
    # dataset = PolygonAEAnimalDataset(r'D:\projects\shape\shape_representation_analysis\polygon_animal_dataset.csv',
    #                                  r'D:\projects\shape\shape_representation_analysis\polygon_animal_dataset_label.csv',
    #                                  twoDim=True)
    # dataiter = iter(dataset)
    # data, label = next(dataiter)
    # print(data)
    # print(label)
    # plt.scatter(data[0, :], data[1, :])
    # plt.show()
    #
    # # x = torch.tensor([[1,2],[3,4]]).float()
    # # y = torch.tensor([[1,2],[3,5]]).float()
    # # c = torch.nn.MSELoss()
    # # loss = c(x,y)
    # # print(loss)

    ############ Random Rotate ############
    rt = RandomRotatePoints(20)
    hft = RandomFlipPoints(0.5)
    vft = RandomFlipPoints(0.2, True)
    ir = IndexRotate()
    transforms = torchvision.transforms.Compose([hft, vft, rt, ir])
    dataset = PolygonAEAnimalDataset(r"D:\projects\shape\shape_representation_analysis\polygon_animal_dataset.csv",
                                     r"D:\projects\shape\shape_representation_analysis\polygon_animal_dataset_label.csv",
                                     transforms,
                                     True)
    it = iter(dataset)
    sample, label = next(it)
    print(sample.shape)
    plt.plot(sample[0, :], sample[1, :])
    for i in range(sample.shape[1]):
        plt.annotate(i, (sample[0, i], sample[1, i]))
    plt.show()
