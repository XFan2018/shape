"""
extract shapes (32 polygon coordinates) from animal dataset
generate a dataset for dictionary learning
"""
import sys

sys.path.append("D:\\projects\\shape\\sparse_coding")
from sparse_coding import *
import argparse
import torchvision
import torch
from image_to_polygon2 import *  # polygon coordinates normalize to 0 mean and unit norm
from Animal_dataset2 import *
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser(description="extract polygon coordinates from silhouettes")
parser.add_argument("-dst", "--dataset", help="path to the dataset")
parser.add_argument("-ext", "--extension", help="path to the dataset")
parser.add_argument("-pn", "--polygon_number", help="number of polygon coordinates")
args = parser.parse_args()


def get_shapes():
    shapes = []
    transforms = torchvision.transforms.Compose(
        [PolygonTransform(int(args.polygon_number))])  # transform to flatten polygon coordinates tensor
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, inputs in enumerate(dataloader):
        #print(inputs[0].shape)  # to delete
        #shapes.append(np.hstack(np.array(inputs[0][0])))  # to delete
        shapes.append([np.hstack((np.array(inputs[0][0]), np.array(inputs[1][0])))])
    shapes = np.vstack(shapes)
    a = {"shapes": shapes[:, 0:-1], "target": shapes[:, -1]}  # remove the comment
    # a = {"shapes": shapes}  # to delete
    sio.savemat("sorted_shapes-120.mat", a)


if __name__ == "__main__":
    get_shapes()
