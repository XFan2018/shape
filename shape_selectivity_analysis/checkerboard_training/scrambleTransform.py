import os

import numpy as np
import sys
sys.path.append(r"D:\projects\shape")
from shape_selectivity_analysis.checkerboard_training.scrambleImage import scramble_image, scramble_image_row
import random
random.seed(os.getenv("SEED"))

class ScrambleTransform(object):
    def __init__(self, block_size):
        # assume the block is a square
        self.block_size = block_size

    def __call__(self, sample):
        """
        :param sample:  PIL image
        :return:        dictionary
        """
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        sample = np.transpose(sample, (2, 0, 1))  # (channel, row, col)
        result = scramble_image(sample, self.block_size, self.block_size)
        result = np.transpose(result, (1, 2, 0))  # (row, col, channel)
        return result


class HorizontalScrambleTransform(object):
    def __init__(self, block_size):
        # assume the block is a square
        self.block_size = block_size

    def __call__(self, sample):
        """
        :param sample:  PIL image
        :return:        dictionary
        """
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        sample = np.transpose(sample, (2, 0, 1))  # (channel, row, col)
        result = scramble_image_row(sample, self.block_size, self.block_size)
        result = np.transpose(result, (1, 2, 0))  # (row, col, channel)
        return result


class ScrambleRandomTransform(object):
    def __init__(self):
        self.block_size = [32, 56, 112]

    def __call__(self, sample):
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        sample = np.transpose(sample, (2, 0, 1))  # (channel, row, col)
        block_size = random.choice(self.block_size)
        result = scramble_image(sample, block_size, block_size)
        result = np.transpose(result, (1, 2, 0))  # (row, col, channel)
        return result
