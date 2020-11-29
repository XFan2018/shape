from image_to_polygon import PolygonTransform, TurningAngleTransform, FourierDescriptorTransform
from Hemera_dataset import HemeraDataset
from Animal_dataset import AnimalDataset
import torchvision
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

polygon_number = 128
dataset_path = r"D:\projects\shape_dataset\Hemera"
extension = "png"
batch_size = 1
file_name = "Fourier_descriptor_hemera_dataset_128.csv"
# file_label_name = "Fourier_descriptor_animal_dataset_label_validation.csv"

polygon_transform = torchvision.transforms.Compose([PolygonTransform(polygon_number, oneDim=True)])
turning_angle_transform = torchvision.transforms.Compose([TurningAngleTransform(polygon_number)])
Fourier_descriptor_transform = torchvision.transforms.Compose([FourierDescriptorTransform(polygon_number)])


training_set = HemeraDataset(dataset_path, extension, transforms=Fourier_descriptor_transform)
dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)


def extract_data_from_silhouette():
    result = []
    for data_index, (inputs, label) in enumerate(dataloader):
        print(data_index)
        result.append(inputs[0])
    result = np.vstack(result)
    np.savetxt(file_name, result, delimiter=",")


def extract_data_from_silhouette_with_label():
    training_set = AnimalDataset(dataset_path, extension, transforms=turning_angle_transform)
    csvfile = open(file_name, "w")
    csvfilelabel = open(file_label_name, "w")
    result = []
    label_list = []
    for data_index, (inputs, label) in enumerate(training_set):
        print(data_index)
        result.append(inputs)
        label_list.append(label)
    label_list = np.array(label_list, dtype=int)
    result = np.vstack(result)
    print(result.shape)
    np.savetxt(csvfile, result, delimiter=",")
    np.savetxt(csvfilelabel, label_list, delimiter=",")
    csvfile.close()
    csvfilelabel.close()


extract_data_from_silhouette()
# train = pd.read_csv(file_name)
# print(train.values.shape)
# x = train.values[0]
# x = x.reshape(2, polygon_number).transpose()
# plt.scatter(x[:, 0], x[:, 1])
# for i in range(len(x)):
#     plt.annotate(i, (x[i, 0], x[i, 1]))
# plt.show()

train = pd.read_csv(file_name, header=None)
# label = pd.read_csv(file_label_name, header=None)
print(train.values.shape)
# print(label.values.shape)
