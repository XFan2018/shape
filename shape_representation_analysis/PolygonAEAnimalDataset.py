import sys

import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from shape_representation_analysis.shape_transforms import RandomRotatePoints, RandomFlipPoints, IndexRotate, \
    WhiteNoise, LowPassNoise
import os
np.random.seed(os.getenv("SEED"))

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
    rrt = RandomRotatePoints(20)
    hft = RandomFlipPoints(0.5)
    vft = RandomFlipPoints(0.2, True)
    irt = IndexRotate()
    # rtt = RandomTranslation()
    cgn = LowPassNoise(-1.5, 1)
    transforms = torchvision.transforms.Compose([hft, rrt, irt])
    # transforms = torchvision.transforms.Compose([])
    dataset = PolygonAEAnimalDataset(r"D:\projects\shape\shape_representation_analysis\polygon_animal_dataset.csv",
                                     r"D:\projects\shape\shape_representation_analysis\polygon_animal_dataset_label.csv",
                                     transforms,
                                     True)
    it = iter(dataset)
    try:
        count = 0
        while count < 1:
            fig = plt.figure(count, (20, 20), facecolor="w", edgecolor="b")
            sample, label = next(it)
            print(count)
            # plt.plot(sample[0, :], sample[1, :], "o-")
            # plt.show()

            contour_complex = np.zeros(sample.shape[1], dtype=complex)
            contour_complex.real = sample[0, :]
            contour_complex.imag = sample[1, :]
            fft_num = 32
            fd = np.fft.fft(contour_complex, fft_num)
            fd.real = np.roll(fd.real, fft_num // 2)
            fd.imag = np.roll(fd.imag, fft_num // 2)
            a = -1.5
            b = 2.5
            ax1 = plt.subplot(241)
            ax2 = plt.subplot(242)
            ax3 = plt.subplot(246)
            ax4 = plt.subplot(245)
            ax5 = plt.subplot(243)
            ax6 = plt.subplot(247)
            ax7 = plt.subplot(244)
            ax8 = plt.subplot(248)
            ax1.title.set_text("no lpn")
            ax2.title.set_text("real")
            ax5.title.set_text("real + noise")
            ax3.title.set_text("imaginary")
            ax6.title.set_text("imaginary + noise")
            ax7.title.set_text("real noise")
            ax8.title.set_text("imaginary noise")
            ax4.title.set_text("lpn")
            ax2.set_ylabel("magnitude")
            ax6.set_xlabel("frequency")
            ax3.set_xlabel("frequency")
            ax3.set_ylabel("magnitude")
            ax1.plot(sample[0, :], sample[1, :], "o-")
            for i in range(32):
                ax1.annotate(i, (sample[0, i], sample[1, i]))
            ax2.plot(list(range(-fft_num // 2, fft_num // 2)), fd.real, "o-")
            for i in range(fft_num):
                ax2.annotate(i-16, (i-16, fd.real[i] + 0.5))
            ax3.plot(list(range(-fft_num // 2, fft_num // 2)), fd.imag, "o-")
            for i in range(fft_num):
                ax3.annotate(i-16, (i-16, fd.imag[i] + 0.5))
            noise = np.zeros(fd.shape, dtype=complex)
            k = np.array(list(reversed(range(1, 17))) + [1] + list(range(1, 16)))
            noise_real = [np.random.normal(0.0, b * pow(i, a), size=1) for i in k]
            print("noise_real ", noise_real)
            noise_imaginary = [np.random.normal(0.0, b * pow(i, a), size=1) for i in k]
            print("noise_imag ", noise_imaginary)
            noise.real = np.concatenate(noise_real)
            noise.imag = np.concatenate(noise_imaginary)
            print("k: ", k)
            noise.real[16] = 0
            noise.imag[16] = 0
            fd_noise = fd + noise

            ax5.plot(list(range(-fft_num // 2, fft_num // 2)), fd_noise.real, "o-")
            for i in range(fft_num):
                ax5.annotate(i-16, (i-16, fd_noise.real[i] + 0.5))
            ax6.plot(list(range(-fft_num // 2, fft_num // 2)), fd_noise.imag, "o-")
            for i in range(fft_num):
                ax6.annotate(i-16, (i-16, fd_noise.imag[i] + 0.5))

            ax7.plot(list(range(-fft_num // 2, fft_num // 2)), noise.real, "o-")
            for i in range(fft_num):
                ax5.annotate(i-16, (i-16, fd_noise.real[i] + 0.5))
            ax8.plot(list(range(-fft_num // 2, fft_num // 2)), noise.imag, "o-")
            for i in range(fft_num):
                ax6.annotate(i-16, (i-16, fd_noise.imag[i] + 0.5))
            # rfd = np.fft.ifft(np.concatenate([fd[0:8], fd[24:]]))
            fd_noise.real = np.roll(fd_noise.real, fft_num // 2)
            fd_noise.imag = np.roll(fd_noise.imag, fft_num // 2)
            rfd = np.fft.ifft(fd_noise)
            # plt.xlim([-1.25, 1.25])
            # plt.ylim([-1.25, 1.25])
            # plt.xticks(fontsize=20)
            # plt.yticks(fontsize=20)
            ax4.plot(rfd.real, rfd.imag, "o-")
            for i in range(32):
                ax4.annotate(i, (rfd.real[i], rfd.imag[i]))
            # plt.savefig(rf"./images/img-{count}.jpg")
            plt.show()
            count += 1
    except StopIteration:
        sys.exit()
