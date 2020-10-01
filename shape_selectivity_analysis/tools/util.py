import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def unpickle(file):
    """
    :param file: dataset path
    :return: dataset dictionary
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getImage(file, i):
    """
    :param file: cifar-100 dataset path
    :param i: data index
    :return:  np-img and its label
    """
    d = unpickle(file)
    X = d[b'data']
    Y = d[b'fine_labels']
    X = np.reshape(X, (50000, 3, 32, 32))
    Y = np.array(Y)
    return X[i], Y[i]


def showImage(file, i):
    """
    :param file: dataset path
    :param i: image index
    :return: label and show image
    """
    img, label = getImage(file, i)
    plt.imshow(np.transpose(img, (1, 2, 0)))  # img (channel, row, col)
    print("label", label)
    plt.show()


def show_image(img):
    """
    :param img: image returned from DataLoader
                batch images require to be preprocessed by make_grid
    """
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def show_multi_images(images):
    """
    :param images: image returned from DataLoader

    """
    print("image size: ", images.size())
    result = make_grid(images, padding=5)
    show_image(result)
