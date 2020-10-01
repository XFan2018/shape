from PIL import Image
from project3.sparse_coding import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
from project3.Animal_dataset import AnimalDataset


class RasterPoints(object):
    def __init__(self, image_dim=224, fill=True, AA_multiplier=5):
        assert isinstance(image_dim, int)
        assert isinstance(fill, bool)
        self.image_dim = image_dim
        self.fill = fill
        self.AA_multiplier = AA_multiplier

    def __call__(self, points):
        image_dim = self.image_dim
        fill = self.fill
        # for antialiasing
        AA_multiplier = self.AA_multiplier
        image_dim = AA_multiplier * image_dim
        points = points * image_dim / 2 + image_dim / 2
        image = Image.new("L", (image_dim, image_dim), color=0)
        draw = ImageDraw.Draw(image)
        points = np.append(points, np.reshape(points[0], (1, 2)), axis=0)
        points = tuple(map(tuple, points))
        draw.line((points), fill=255, width=AA_multiplier)
        if fill:
            draw.polygon((points), fill=255)
        image = image.rotate(180)
        image = ImageOps.mirror(image)
        image_dim = int(image_dim / AA_multiplier)
        image_new = image.resize((image_dim, image_dim), resample=Image.BILINEAR)
        return np.asarray(image_new)


class RasterPointsTransformation(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.rp = RasterPoints()

    def __call__(self, image):
        image = np.array(image)  # PIL image to numpy (row, col, channel)
        polygon = im2poly(image, n_samples=self.n_samples, normalized=False)
        # normalization
        polygon -= polygon.mean(axis=0)
        polygon /= np.max(np.abs(polygon))
        polygon *= 0.9
        # ensure area are the same
        # polygon = self.match_area_points(polygon)
        #############calculate the area###################################################
        # x_i = polygon[0, :-1]
        # x_i_plus_1 = polygon[0, 1:]
        # y_i = polygon[1, :-1]
        # y_i_plus_1 = polygon[1, 1:]
        # dataset_area = np.abs(0.5 * np.sum(x_i * y_i_plus_1 - y_i * x_i_plus_1))
        # print(dataset_area)
        ##################################################################################
        polygon = self.rp(polygon)
        im = Image.fromarray(polygon)
        im = im.convert("RGB")
        return im



# im = Image.open("D:\\projects\\summerProject2020\\project3\\animal_silhouette_training\\dolphine\\dolphine2.tif")
# im.show()
# polygon = im2poly(im, 32)
# polygon = np.flip(polygon, axis=0)
# plt.scatter(polygon[:, 0], polygon[:, 1])
# for i, p in enumerate(polygon):
#     plt.annotate(i, (p[0], p[1]))
# plt.show()
# rpt = RasterPointsTransformation(32)
# im = rpt(im)
# im.show()
