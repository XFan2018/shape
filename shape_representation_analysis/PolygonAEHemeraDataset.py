import pandas as pd
import torch
import matplotlib.pyplot as plt


class PolygonAEHemeraDataset():
    """
    Silhouette Polygon dataset
    """

    def __init__(self, root_dir, polygon_number, transforms=None, twoDim=False):
        self.root_dir = root_dir
        self.transforms = transforms
        samples = pd.read_csv(root_dir).values
        self.samples = samples
        self.twoDim = twoDim
        self.polygon_number = polygon_number

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.twoDim:
            sample = sample.reshape(2, self.polygon_number)
        sample = sample.copy()
        sample = torch.tensor(sample)
        return sample


class TurningAngleAEHemeraDataset():
    """
    Silhouette Turning Angle dataset
    """

    def __init__(self, root_dir, polygon_number, transforms=None, twoDim=False):
        self.root_dir = root_dir
        self.transforms = transforms
        samples = pd.read_csv(root_dir).values
        self.samples = samples
        self.twoDim = twoDim
        self.polygon_number = polygon_number

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.twoDim:
            sample = sample.reshape(1, self.polygon_number)
        sample = sample.copy()
        sample = torch.tensor(sample)
        return sample


class FourierDescriptorAEHemeraDataset():
    """
    Silhouette Fourier Descriptor dataset
    """

    def __init__(self, root_dir, polygon_number, transforms=None, twoDim=False):
        self.root_dir = root_dir
        self.transforms = transforms
        samples = pd.read_csv(root_dir).values
        self.samples = samples
        self.twoDim = twoDim
        self.polygon_number = polygon_number

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.twoDim:
            sample = sample.reshape(1, self.polygon_number * 2)
        sample = sample.copy()
        sample = torch.tensor(sample)
        return sample


if __name__ == "__main__":
    dataset = PolygonAEHemeraDataset(r'D:\projects\shape\shape_representation_analysis\polygon_hemera_training.csv', twoDim=True)
    dataiter = iter(dataset)
    data = next(dataiter)
    data = next(dataiter)
    print(data)
    plt.scatter(data[0, :], data[1, :])
    plt.show()
    x = torch.tensor([[1,2],[3,4]]).float()
    y = torch.tensor([[1,2],[3,5]]).float()
    c = torch.nn.MSELoss()
    loss = c(x,y)
    print(loss)

    ############ Random Rotate ############

