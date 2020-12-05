import random
import os
from settings import *
import shutil

selected_cate = ["bear", "bicycle", "boat", "bottle", "cat", "car", "clock", "keyboard"]


def random_sample_images_into_four_blocksize():
    for cate in os.listdir(IMAGENET_16CAT_DIR):
        if cate in selected_cate:
            img_arr = os.listdir(os.path.join(IMAGENET_16CAT_DIR, cate))
            blocksize8 = random.sample(img_arr, 25)
            print(blocksize8)
            for img in blocksize8:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize8", cate))
            img_arr = list(filter(lambda v: v not in blocksize8, img_arr))
            blocksize16 = random.sample(img_arr, 25)
            print(blocksize16)
            for img in blocksize16:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize16", cate))
            img_arr = list(filter(lambda v: v not in blocksize16, img_arr))
            blocksize28 = random.sample(img_arr, 25)
            print(blocksize28)
            for img in blocksize28:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize28", cate))
            img_arr = list(filter(lambda v: v not in blocksize28, img_arr))
            blocksize56 = random.sample(img_arr, 25)
            print(blocksize56)
            for img in blocksize56:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize56", cate))
            set_blocksize8 = set(blocksize8)
            set_blocksize16 = set(blocksize16)
            set_blocksize28 = set(blocksize28)
            set_blocksize56 = set(blocksize56)
            if len(set_blocksize8.intersection(set_blocksize16).
                           union(set_blocksize8.intersection(set_blocksize28)).
                           union(set_blocksize8.intersection(set_blocksize56)).
                           union(set_blocksize16.intersection(set_blocksize28)).
                           union(set_blocksize16.intersection(set_blocksize56)).
                           union(set_blocksize28).intersection(set_blocksize56)) != 0:
                raise ValueError
            print("-----------------")



