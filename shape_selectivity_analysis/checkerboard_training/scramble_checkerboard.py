import torch
import torchvision
from PIL import Image
from shape_selectivity_analysis.checkerboard_training.scrambleImage import scramble_image, scramble_image_row
import torchvision.transforms as transforms
import numpy as np
import random

im1 = Image.open(
    r"D:\projects\shape\shape_selectivity_analysis\checkerboard_training\ILSVRC2012_val_00040753.JPEG")
im2 = Image.open(
    r"D:\projects\shape\shape_selectivity_analysis\checkerboard_training\ILSVRC2012_val_00040871.JPEG")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(256),
     torchvision.transforms.CenterCrop(224)])

# resize image to 224
im1 = transform(im1)
im2 = transform(im2)
im1 = np.array(im1)
im2 = np.array(im2)


def checkerboard(im1: np.ndarray, im2: np.ndarray, size: int, horizontal_only=False, use_lattice=False):
    """
    :param im1: ground truth image （numpy）
    :param im2: adversarial attack image (numpy)
    :param size: checkerboard block sizes
    :param horizontal_only: scrambled horizontally
    :return: checkerboard image that combines ground truth image and scrambled adversarial attack image (PIL)
    """
    #  scramble adversarial image
    transform = transforms.ToPILImage()
    im1 = transform(im1)
    im2 = np.transpose(im2, (2, 0, 1))
    if horizontal_only:
        im2 = scramble_image_row(im2, size, size)
    else:
        im2 = scramble_image(im2, size, size)
    im2 = np.transpose(im2, (1, 2, 0))
    im2 = Image.fromarray(im2)
    # create new image and pixel_map (original image remain unchanged)
    pixel_map1 = im1.load()
    pixel_map2 = im2.load()
    im1_new = Image.new(im1.mode, im1.size)
    pixel_map1_new = im1_new.load()
    im2_new = Image.new(im1.mode, im2.size)
    pixel_map2_new = im2_new.load()
    # assign pixel values so that they are all identical to the original images
    for i in range(im1_new.size[0]):
        for j in range(im1.size[1]):
            pixel_map1_new[i, j] = pixel_map1[i, j]

    for i in range(im2_new.size[0]):
        for j in range(im2.size[1]):
            pixel_map2_new[i, j] = pixel_map2[i, j]

    # make checkerboard
    mod = im1.size[0] % size
    if mod != 0:
        raise Exception("cannot make checker board with size: " + str(size))
    number = im1.size[0] // size
    rand = random.random()
    # print(rand)
    lattice = 1
    for i in range(0, number):  # row of the checkerboard
        for j in range(0, number, 2):  # col of the checkerboard to be replace by the second image
            if i % 2 == 0:
                k = j
                j += 1
            else:
                k = j + 1
            if use_lattice:
                if k != number:
                    for m in range(i * size, (i + 1) * size):
                        print(k)
                        for n in range(k * size, (k + 1) * size):
                            is_boundary = (m < (i * size) + lattice or m > (i + 1) * size - 1 - lattice or n < (
                                        k * size) + lattice or n > (k + 1) * size -1 - lattice)
                            if rand >= 0.5:
                                if is_boundary:
                                    pixel_map1_new[m, n] = (128, 128, 128)
                            else:
                                if is_boundary:
                                    pixel_map2_new[m, n] = (128, 128, 128)
            if j == number:
                continue
            for m in range(i * size, (i + 1) * size):
                for n in range(j * size, (j + 1) * size):
                    if use_lattice:
                        is_boundary = (m < (i * size) + lattice or m > (i + 1) * size -1 - lattice or n < (j * size) + lattice or n > (j + 1) * size -1 - lattice)
                    else:
                        is_boundary = False
                    if rand >= 0.5:
                        if is_boundary:
                            pixel_map1_new[m, n] = (128, 128, 128)
                        else:
                            pixel_map1_new[m, n] = pixel_map2_new[m, n]
                    else:
                        if is_boundary:
                            pixel_map2_new[m, n] = (128, 128, 128)
                        else:
                            pixel_map2_new[m, n] = pixel_map1_new[m, n]



    if rand >= 0.5:
        return im1_new
    else:
        return im2_new


def checkerboard_intact_gray(im1, size):
    """
    :param im1: intadt image (numpy)
    :param size: checker board block sizes
    :return: checker board image that combines gray images and intact image (PIL)
    """
    #  scramble adversarial image
    transform = transforms.ToPILImage()
    im1 = transform(im1)
    # create new image and pixel_map (original image remain unchanged)
    pixel_map1 = im1.load()
    im1_new = Image.new(im1.mode, im1.size)
    pixel_map1_new = im1_new.load()
    # assign pixel values so that they are all identical to the original images
    for i in range(im1_new.size[0]):
        for j in range(im1.size[1]):
            pixel_map1_new[i, j] = pixel_map1[i, j]

    # make checker board
    mod = im1.size[0] % size
    if mod != 0:
        raise Exception("cannot make checker board with size: " + str(size))
    number = im1.size[0] // size
    rand = random.random()
    # print(rand)
    for i in range(0, number):  # row of the checker board
        for j in range(0, number, 2):  # col of the checker board to be replace by the second image
            if rand >= 0.5:
                if i % 2 == 0:
                    j += 1
            else:
                if i % 2 == 1:
                    j += 1
            if j == number:
                continue
            for m in range(i * size, (i + 1) * size):
                for n in range(j * size, (j + 1) * size):
                    pixel_map1_new[m, n] = (128, 128, 128)

    return im1_new


def checkerboard_scrambled_gray(im2, size, horizontal_only=False):
    """
    :param im2: scrambled image (numpy)
    :param size: checker board block sizes
    :param horizontal_only: scramble horizontal only
    :return: checker board image that combines gray images and scrambled image (PIL)
    """
    #  scramble adversarial image
    transform = transforms.ToPILImage()
    im2 = transform(im2)
    im2 = np.array(im2)
    im2 = np.transpose(im2, (2, 0, 1))
    if horizontal_only:
        im2 = scramble_image_row(im2, size, size)
    else:
        im2 = scramble_image(im2, size, size)
    im2 = np.transpose(im2, (1, 2, 0))
    im2 = Image.fromarray(im2)
    # create new image and pixel_map (original image remain unchanged)
    pixel_map2 = im2.load()
    im2_new = Image.new(im2.mode, im2.size)
    pixel_map2_new = im2_new.load()
    # assign pixel values so that they are all identical to the original images

    for i in range(im2_new.size[0]):
        for j in range(im2.size[1]):
            pixel_map2_new[i, j] = pixel_map2[i, j]

    # make checker board
    mod = im2.size[0] % size
    if mod != 0:
        raise Exception("cannot make checker board with size: " + str(size))
    number = im2.size[0] // size
    rand = random.random()
    # print(rand)
    for i in range(0, number):  # row of the checker board
        for j in range(0, number, 2):  # col of the checker board to be replace by the second image
            if rand >= 0.5:
                if i % 2 == 0:
                    j += 1
            else:
                if i % 2 == 1:
                    j += 1
            if j == number:
                continue
            for m in range(i * size, (i + 1) * size):
                for n in range(j * size, (j + 1) * size):
                    pixel_map2_new[m, n] = (128, 128, 128)

    return im2_new


def checkerboard_batch(im1_batch, im2_batch, block_size, horizontal_only=False, use_lattice=False):
    """
    :param im1_batch: a batch of numpy images (ground truth)
    :param im2_batch: a batch of numpy images (adversary)
    :param block_size: see above
    :param horizontal_only: see above
    :return: a batch of tensor images (checker board)
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    result = []
    length = len(im1_batch)
    for i in range(length):
        cb = checkerboard(im1_batch[i], im2_batch[i], block_size, horizontal_only, use_lattice)
        # cb.show()
        cb = np.array(cb)
        cb = transform(cb)
        cb = cb.unsqueeze(0)
        result.append(cb)
    result = torch.cat(result)
    return result


def checkerboard_intact_gray_batch(im1_batch, block_size):
    """
    :param im1_batch: a batch of numpy images (ground truth)
    :param block_size: see above
    :return: a batch of tensor images (checker board)
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    result = []
    length = len(im1_batch)
    for i in range(length):
        cb = checkerboard_intact_gray(im1_batch[i], block_size)
        # cb.show()
        cb = np.array(cb)
        cb = transform(cb)
        cb = cb.unsqueeze(0)
        result.append(cb)
    result = torch.cat(result)
    return result


def checkerboard_scrambled_gray_batch(im2_batch, block_size, horizontal_only=False):
    """
    :param im2_batch: a batch of numpy images (scrambled)
    :param block_size: see above
    :param horizontal_only: see above
    :return: a batch of tensor images (checker board)
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    result = []
    length = len(im2_batch)
    for i in range(length):
        cb = checkerboard_scrambled_gray(im2_batch[i], block_size, horizontal_only)
        # cb.show()
        cb = np.array(cb)
        cb = transform(cb)
        cb = cb.unsqueeze(0)
        result.append(cb)
    result = torch.cat(result)
    return result


if __name__ == "__main__":
    # im1_new = checkerboard(im1, im2, 56, True, True)
    # transform1 = transforms.ToTensor()
    # transform2 = transforms.ToPILImage()
    # im1 = transform1(im1)
    # im1 = transform2(im1)
    #
    # im1_new.show()
    pass