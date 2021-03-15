import skimage.util as sku
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
torch.manual_seed(int(os.getenv("SEED")))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(int(os.getenv("SEED")))


def split_image_into_blocks(img, xdim=1, ydim=1):
    """
    :param img:  np-img (channel, row, col)
    :param xdim: block row shape
    :param ydim: block col shape
    :return:     blocks
    Note: be careful, it can return None
    """
    z, x, y = img.shape
    if not (x % xdim == 0 and y % ydim == 0):
        print("dim arguments cannot divide the image")
        return

    blocks = sku.view_as_blocks(img, (3, xdim, ydim))
    blocks = np.squeeze(blocks, axis=0)
    return blocks


def rebuild_splited_image(blocks):
    """
    :param blocks: blocks returned from splitImageIntoBlocks (x/xdim, y/ydim, channel#, xdim, ydim)
    :return: rebuilt image (channel, row, col)
    """

    blocks_num_per_col = blocks.shape[0]
    row_num_per_block = blocks.shape[3]
    img = blocks.transpose(2, 0, 3, 1, 4).reshape(3, blocks_num_per_col * row_num_per_block, -1)
    return img


def scramble_blocks(blocks):
    """
    :param blocks: split image blocks
    :return: shuffled split image blocks
    """
    blocks_num_per_col = blocks.shape[0]
    blocks_num_per_row = blocks.shape[1]
    shuffled_row_arr = np.arange(0, blocks_num_per_col)
    shuffled_col_arr = np.arange(0, blocks_num_per_row)
    np.random.shuffle(shuffled_col_arr)
    np.random.shuffle(shuffled_row_arr)
    shuffled_row = []
    shuffled_col = []
    for i in shuffled_row_arr:
        for j in shuffled_col_arr:
            shuffled_col.append(blocks[i][j])
        shuffled_row.append(shuffled_col)
        shuffled_col = []
    result = np.array(shuffled_row)
    return result


def scramble_blocks2(blocks):
    """
    :param blocks: split image blocks
    :return: shuffled split image blocks
    """
    blocks_num_per_col = blocks.shape[0]
    blocks_num_per_row = blocks.shape[1]
    shuffled_arr = np.arange(0, blocks_num_per_col * blocks_num_per_row)
    np.random.shuffle(shuffled_arr)
    shuffled_row = []
    shuffled_col = []
    for i in shuffled_arr:
        col = i // blocks_num_per_row
        row = i % blocks_num_per_row
        shuffled_row.append(blocks[row][col])
        if len(shuffled_row) == blocks_num_per_row:
            shuffled_col.append(shuffled_row)
            shuffled_row = []
    result = np.array(shuffled_col)
    return result


def scramble_row(blocks):
    """
    :param blocks: split image blocks
    :return: shuffled split image blocks
    """
    blocks_num_per_col = blocks.shape[0]
    blocks_num_per_row = blocks.shape[1]
    shuffled_row = []
    shuffled_col = []
    for i in range(blocks_num_per_row):
        shuffled_arr = np.arange(0, blocks_num_per_col)
        np.random.shuffle(shuffled_arr)
        for j in shuffled_arr:
            shuffled_row.append(blocks[i][j])
            if len(shuffled_row) == blocks_num_per_row:
                shuffled_col.append(shuffled_row)
                shuffled_row = []
    result = np.array(shuffled_col)
    return result


def scramble_image(img, xdim, ydim):
    blocks = split_image_into_blocks(img, xdim, ydim)
    blocks = scramble_blocks2(blocks)
    return rebuild_splited_image(blocks)


def scramble_image_row(img, xdim, ydim):
    blocks = split_image_into_blocks(img, xdim, ydim)
    blocks = scramble_row(blocks)
    return rebuild_splited_image(blocks)


# f = "cifar-100-python/train"  # dataset path
# img, label = getImage(f, 100)  # 2nd arg is the index of the dataset
# b = scramble_image(img, 8, 8)  # img: (channel, row, col)
# result = np.transpose(b, (1, 2, 0))  # convert to normal dimension
# img = np.transpose(img, (1, 2, 0))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title("Origin")
#
#
# plt.subplot(1, 2, 2)
# plt.title("Scrambled")
# plt.imshow(result)
#
# plt.show()
#
if __name__ == "__main__":
    img = Image.open(
        "D:\\projects\\summerProject2020\\project1\\imagenet_val_testing_dataset\\n01440764\\ILSVRC2012_val_00031333.JPEG").convert(
        "RGB")
    # print(img.size)
    img = img.resize((224, 224))
    # plt.imshow(img)
    # plt.show()
    # print(img.size)
    img = np.transpose(img, (2, 0, 1))
    # print(type(img))
    # list = [2, 5, 10, 20, 40, 50, 100]
    # for size in list:
    im = scramble_image_row(img, 32, 32)
    im = np.transpose(im, (1, 2, 0))
    #     im = Image.fromarray(im)
    #     im.save("img_block_size_"+str(size)+".png")
    plt.imshow(im)
    plt.show()
