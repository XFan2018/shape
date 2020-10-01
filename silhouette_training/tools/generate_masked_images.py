import PIL
import numpy as np
from PIL import Image
import torchvision
import torch

file_path = "/Users/leo/PycharmProjects/summerProject/project2/fish.JPEG"
img = Image.open(file_path)


def mask_patch(img, patch_size):
    """
    :param img: image
    :return:
             patch size = 20
             a list of 25 + 1 + 1 = 27 images will be returned (type = <class 'PIL.Image.Image'>)
             1st "1" is the image masked with all patches simultaneously, 2nd "1" is the original image
    """
    patch_size_list = [20, 30, 40, 50, 60, 70]
    if patch_size not in patch_size_list:
        raise Exception("Sorry, no patch size must be one of 20, 30, 40, 50, 60, 70 numbers")
    switcher = {
        20: {
            "offset": 22,
            "interval": 20 * 2,
            "img_number": 26
        },

        30: {
            "offset": 7,
            "interval": 30 * 2,
            "img_number": 17
        },

        40: {
            "offset": 12,
            "interval": 40 * 2,
            "img_number": 10
        },

        50: {
            "offset": 37,
            "interval": 50 * 2,
            "img_number": 5
        },

        60: {
            "offset": 22,
            "interval": 60 * 2,
            "img_number": 5
        },

        70: {
            "offset": 7,
            "interval": 70 * 2,
            "img_number": 5
        }

    }
    list_img = []  # a list of 27 images, see return for details
    list_img_pixel_map = []  # 27 pixel maps for the 27 images
    img = img.resize((224, 224))  # original image is re-sized to 224
    pixel_map = img.load()
    for _ in range(switcher[patch_size]["img_number"]):
        # create n images and pixel maps from original images
        new_img = Image.new(img.mode, img.size)
        pixel_map_new = new_img.load()

        # assign pixel values so that they are all identical to the original images
        for i in range(new_img.size[0]):
            for j in range(new_img.size[1]):
                pixel_map_new[i, j] = pixel_map[i, j]

        # append them to the list
        list_img.append(new_img)
        list_img_pixel_map.append(pixel_map_new)

    # create iterator for the two lists (images and pixel maps)
    iter_img_pixel_map = iter(list_img_pixel_map)

    # generate the masks location
    list_offset_row = range(switcher[patch_size]["offset"], 224 - switcher[patch_size]["offset"],
                            switcher[patch_size]["interval"])
    list_offset_col = range(switcher[patch_size]["offset"], 224 - switcher[patch_size]["offset"],
                            switcher[patch_size]["interval"])
    for i in list_offset_col:
        print(i)
    for offset_row in list_offset_row:
        for offset_col in list_offset_col:

            # image masked with individual patch
            pixel_map_individual_patch = next(iter_img_pixel_map)

            # image masked with all patches simultaneously
            pixel_map_all_patch = list_img_pixel_map[switcher[patch_size]["img_number"] - 1]
            for i in range(offset_row, offset_row + patch_size):
                for j in range(offset_col, offset_col + patch_size):
                    pixel_map_individual_patch[i, j] = (0, 0, 0)
                    pixel_map_all_patch[i, j] = (0, 0, 0)

    # add original image to the end of the list
    list_img.append(img)
    return list_img


result = mask_patch(img, 20)
for im in result:
    im.show()
