from sparse_coding2 import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#im = Image.open("/Users/leo/PycharmProjects/summerProject/project2/animal_silhouette_testing/bird/bird8.tif")


# # im = im.convert("RGB")


class PolygonTransform(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, sample):
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        result = im2poly(sample, n_samples=self.n_samples)
        result = np.flip(result, axis=0)
        return result


class TurningAngleTransform(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, sample):
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        polygon = im2poly(sample, n_samples=self.n_samples)
        polygon = np.flip(polygon, axis=0)  # counter clockwise
        return self.get_turning_angles(polygon)

    def get_turning_angles(self, outline):
        outline = np.dot(outline, [1, 1j])
        closed = np.concatenate([outline[-1:],
                                 outline,
                                 outline[:1]], axis=0)  # n_samples + 2
        orientations = np.angle(np.diff(closed, axis=0))
        return -self.center_angles(np.diff(orientations, axis=0))

    @staticmethod
    def center_angles(angle):
        return np.angle(np.exp(1j*angle))


# pt = PolygonTransform(120)
# polygons = pt(im)
# print(polygons)
# plt.scatter(polygons[:,0], polygons[:,1])
# plt.show()
# print(len(angles))
# print(np.degrees(angles))

