import os

import numpy as np
import torchvision
import torchvision.transforms as transforms

from settings import *
from shape_selectivity_analysis.checkerboard_training.scrambleTransform import HorizontalScrambleTransform
from shape_selectivity_analysis.checkerboard_training.scramble_checkerboard import checker_board_intact_gray, \
    checker_board

block_sizes = [8, 16, 28, 56]
cate_dict = {
    0: "bear",
    1: "bicycle",
    2: "boat",
    3: "bottle",
    4: "cat",
    5: "car",
    6: "clock",
    7: "keyboard"
}



for x in block_sizes:
    trans_intact = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224)])

    trans_jumbled = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        HorizontalScrambleTransform(x)])

    dataset_intact = torchvision.datasets.ImageFolder(os.path.join(CHECKERBOARD_PREP, f"blocksize{x}"),
                                                      transform=trans_intact)

    dataset_horizontal_jumbled = torchvision.datasets.ImageFolder(os.path.join(CHECKERBOARD_PREP, f"blocksize{x}"),
                                                                  transform=trans_jumbled)

    it_intact = iter(dataset_intact)
    it_jumbled = iter(dataset_horizontal_jumbled)
    list_jumbled = []
    while True:
        try:
            data_jumbled, label_jumbled = next(it_jumbled)
            list_jumbled.append(data_jumbled)
        except StopIteration:
            break
    i = 0
    while True:
        try:
            data_intact, label_intact = next(it_intact)
            data_intact = np.array(data_intact)
            data_jumbled = random.sample(list_jumbled, 1)[0]
            cate = cate_dict[label_intact]
            print(cate)
            img = checker_board(data_intact, data_jumbled, x, True)
            img.save(os.path.join(CHECKERBOARD_DATASET_HUMAN, f"blocksize{x}", cate, f"{cate}{i}.jpeg"))
            i = (i + 1) % 25
        except StopIteration:
            break
            
    while True:
        try:
            data_intact, label_intact = next(it_intact)
            data_intact = np.array(data_intact)
            cate = cate_dict[label_intact]
            print(cate)
            img = checker_board_intact_gray(data_intact, x)
            img.save(os.path.join(CHECKERBOARD_GRAY_DATASET_HUMAN, f"blocksize{x}", cate, f"{cate}{i}.jpeg"))
            i = (i + 1) % 25
        except StopIteration:
            break


