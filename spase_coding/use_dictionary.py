import scipy.io as sio
import numpy as np
from sklearn.decomposition import sparse_encode
from sparse_coding2 import *


# class SparseCodingTransformation(object):
#     def __init__(self, n_samples):
#         self.n_samples = n_samples
#         with open("basisShapesC64L0.1", "rb") as file:
#             self.dictionary = sio.loadmat(file)['component']
#
#     def __call__(self, sample):
#         sample = np.array(sample)  # PIL image to numpy (row, col, channel)
#         # concatenate x and then y coordinates
#         sample = np.hstack((sample[:, 0], sample[:, 1]))
#         result = im2poly(sample, n_samples=self.n_samples)
#         coefficient = sparse_encode(result,
#                                     self.dictionary,
#                                     algorithm="omp",
#                                     n_nonzero_coefs=None,
#                                     alpha=None)
#         return coefficient


with open("basisShapesC64L0.1", "rb") as file:
    dictionary = sio.loadmat(file)['component']

with open("sorted_shapes-32.mat", "rb") as file:
    shapes = sio.loadmat(file)['shapes']
    targets = sio.loadmat(file)['target']
targets = targets.reshape((1700, 1))
print(targets.shape)
coefficients = sparse_encode(shapes,
                             dictionary,
                             algorithm="omp",
                             n_nonzero_coefs=None,
                             alpha=None)
a = {"coefficients": coefficients, "targets": targets}
sio.savemat("coefficients.mat", a)

# for i in np.count_nonzero(coefficients, 1):
#     print(i)
#
# recons = np.dot(coefficients, dictionary)
# errors = np.sum((shapes - recons) ** 2, axis=1)
# print(sum(errors) / len(errors))
