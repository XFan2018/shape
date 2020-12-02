import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

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
    dataset = FourierDescriptorAEHemeraDataset(r'D:\projects\shape\shape_representation_analysis\Fourier_descriptor_hemera_dataset_128.csv', 128, twoDim=False)
    dataiter = iter(dataset)
    data = next(dataiter)
    data = next(dataiter)
    data = next(dataiter)
    data = next(dataiter)
    data = next(dataiter)
    data = next(dataiter)
    print(data)

    output_fd_complex = np.zeros(128, dtype=complex)
    output_fd_complex.real = data[0:128]
    output_fd_complex.imag = data[128:]
    fd_reconstruct = np.fft.ifft(output_fd_complex)
    fd_reconstruct = np.array([fd_reconstruct.real, fd_reconstruct.imag])

    plt.scatter(fd_reconstruct[0, :], fd_reconstruct[1, :])
    plt.show()

    ############ Random Rotate ############

