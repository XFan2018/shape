import os

import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
from PIL import Image
from settings import *
from shape_selectivity_analysis.checkerboard_training.scrambleImage import scramble_image_row
from shape_selectivity_analysis.checkerboard_training.scrambleTransform import HorizontalScrambleTransform
from shape_selectivity_analysis.checkerboard_training.scramble_checkerboard import checkerboard_intact_gray, \
    checkerboard
random.seed(10)
block_sizes = [7, 14, 28, 56]
cate_dict = {
    0: "bear",
    1: "bird",
    2: "boat",
    3: "bottle",
    4: "cat",
    5: "chair",
    6: "dog",
    7: "truck"
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
    i = 0
    while True:
        try:
            data_jumbled, label_jumbled = next(it_jumbled)
            list_jumbled.append((data_jumbled, label_jumbled, i))
            i += 1
            i %= 50
        except StopIteration:
            break
    i = 0
    while True:
        try:
            data_intact, label_intact = next(it_intact)
            data_intact = np.array(data_intact)
            data_jumbled, label_jumbled, j = random.choice(list_jumbled)
            while label_jumbled == label_intact:
                data_jumbled, label_jumbled, j = random.choice(list_jumbled)
            cate = cate_dict[label_intact]
            cate_jumbled = cate_dict[label_jumbled]
            print(cate)
            img = checkerboard(data_intact, data_jumbled, x, True)
            img_intact = Image.fromarray(data_intact)
            temp = np.transpose(data_intact, (2, 0, 1))
            temp = scramble_image_row(temp, x, x)
            temp = np.transpose(temp, (1, 2, 0))
            img_jumbled = Image.fromarray(temp)
            img_gray = checkerboard_intact_gray(data_intact, x)

            img_gray.save(os.path.join(CHECKERBOARD_GRAY_DATASET_HUMAN, f"blocksize{x}", cate, f"{cate}{i}.jpeg"))
            img.save(os.path.join(CHECKERBOARD_DATASET_HUMAN, f"blocksize{x}", cate, f"{cate}{i}-{cate_jumbled}{j}.jpeg"))
            img_intact.save(os.path.join(INTACT_DATASET_HUMAN, f"blocksize{x}", cate, f"{cate}{i}.jpeg"))
            img_jumbled.save(os.path.join(JUMBLED_DATASET_HUMAN, f"blocksize{x}", cate, f"{cate}{i}.jpeg"))
            i = (i + 1) % 50
        except StopIteration:
            break
    # i = 0
    # while True:
    #     try:
    #         data_intact, label_intact = next(it_intact)
    #         data_intact = np.array(data_intact)
    #         cate = cate_dict[label_intact]
    #         print(cate)
    #         img_gray = checker_board_intact_gray(data_intact, x)
    #         img_gray.save(os.path.join(CHECKERBOARD_GRAY_DATASET_HUMAN, f"blocksize{x}", cate, f"{cate}{i}.jpeg"))
    #         i = (i + 1) % 50
    #     except StopIteration:
    #         break
