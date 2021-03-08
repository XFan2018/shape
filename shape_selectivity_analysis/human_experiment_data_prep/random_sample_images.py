import random
import os
from settings import *
import shutil

random.seed(os.getenv("SEED"))
selected_cate = ["bear", "truck", "boat", "bottle", "cat", "bird", "chair", "dog"]


def random_sample_images_into_four_blocksize():
    for cate in os.listdir(IMAGENET_16CAT_DIR):
        if cate in selected_cate:
            img_arr = os.listdir(os.path.join(IMAGENET_16CAT_DIR, cate))
            blocksize7 = random.sample(img_arr, 50)
            print(blocksize7)
            for img in blocksize7:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize7", cate))
            img_arr = list(filter(lambda v: v not in blocksize7, img_arr))
            blocksize14 = random.sample(img_arr, 50)
            print(blocksize14)
            for img in blocksize14:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize14", cate))
            img_arr = list(filter(lambda v: v not in blocksize14, img_arr))
            blocksize28 = random.sample(img_arr, 50)
            print(blocksize28)
            for img in blocksize28:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize28", cate))
            img_arr = list(filter(lambda v: v not in blocksize28, img_arr))
            blocksize56 = random.sample(img_arr, 50)
            print(blocksize56)
            for img in blocksize56:
                shutil.copy(os.path.join(IMAGENET_16CAT_DIR, cate, img),
                            os.path.join(CHECKERBOARD_PREP, "blocksize56", cate))
            set_blocksize7 = set(blocksize7)
            set_blocksize14 = set(blocksize14)
            set_blocksize28 = set(blocksize28)
            set_blocksize56 = set(blocksize56)
            if len(set_blocksize7.intersection(set_blocksize14).
                           union(set_blocksize7.intersection(set_blocksize28)).
                           union(set_blocksize7.intersection(set_blocksize56)).
                           union(set_blocksize14.intersection(set_blocksize28)).
                           union(set_blocksize14.intersection(set_blocksize56)).
                           union(set_blocksize28).intersection(set_blocksize56)) != 0:
                raise ValueError
            print("-----------------")


random_sample_images_into_four_blocksize()
