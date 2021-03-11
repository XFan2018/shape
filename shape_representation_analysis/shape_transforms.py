import random

import torch

import torch_interpolations
from sparse_coding import *

random.seed(os.getenv("SEED"))


class PolygonTransform(object):
    def __init__(self, n_samples, oneDim=False):
        self.n_samples = n_samples
        self.oneDim = oneDim

    def __call__(self, sample):
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        result = im2poly(sample, n_samples=self.n_samples)
        # counter clockwise
        result = np.flip(result, axis=0)
        # index start from the left most point
        idx = np.argmin(result[:, 0])
        new_polygon = np.vstack((result[idx:], result[0:idx]))
        if self.oneDim:
            new_polygon = new_polygon.flatten('F')
        return new_polygon


class EqualArclengthTransform(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, polygon):
        x = np.array(np.concatenate((polygon[:, 0], [polygon[0, 0]]), axis=0))
        y = np.concatenate((polygon[:, 1], [polygon[0, 1]]), axis=0)

        vertices = np.array([x, y]).transpose()

        c = np.dot(vertices, [1, 1j])
        arclength = np.concatenate([[0], np.cumsum(abs(np.diff(c)))])
        target = arclength[-1] * np.arange(self.n_samples) / self.n_samples

        gi_x = torch_interpolations.RegularGridInterpolator([torch.tensor(arclength)], torch.tensor(x))
        gi_y = torch_interpolations.RegularGridInterpolator([torch.tensor(arclength)], torch.tensor(y))

        fx = gi_x([torch.tensor(target)])
        fy = gi_y([torch.tensor(target)])

        return np.concatenate((fx.numpy()[:, np.newaxis], fy.numpy()[:, np.newaxis]), axis=1)


class InterpolationTransform(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, sample):
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        result = im2poly(sample, n_samples=self.n_samples)
        # counter clockwise
        result = np.flip(result, axis=0)
        # index start from the left most point
        idx = np.argmin(result[:, 0])
        new_polygon = np.vstack((result[idx:], result[0:idx]))
        return self.interpolation(new_polygon)

    def interpolation(self, points):
        result = []
        for i in range(self.n_samples):
            result.append(points[i])
            j = (i + 1) % 32
            x = (points[i][0] + points[j][0]) / 2
            y = (points[i][1] + points[j][1]) / 2
            result.append(np.array([x, y]))
        return np.array(result)


class InterpolationTransform2(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, sample):
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        result = im2poly(sample, n_samples=self.n_samples)
        # counter clockwise
        result = np.flip(result, axis=0)
        # index start from the left most point
        idx = np.argmin(result[:, 0])
        new_polygon = np.vstack((result[idx:], result[0:idx]))
        new_polygon = self.interpolation(new_polygon, 32)
        return self.interpolation(new_polygon, 64)

    def interpolation(self, points, points_number):
        result = []
        for i in range(points_number):
            result.append(points[i])
            j = (i + 1) % points_number
            x = (points[i][0] + points[j][0]) / 2
            y = (points[i][1] + points[j][1]) / 2
            result.append(np.array([x, y]))
        return np.array(result)


class FourierDescriptorTransform(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, sample):
        sample = np.array(sample)  # PIL image to numpy (row, col, channel)
        result = im2poly(sample, n_samples=self.n_samples)
        result = np.flip(result, axis=0)
        # index start from the left most point
        idx = np.argmin(result[:, 0])
        result = np.vstack((result[idx:], result[0:idx]))
        contour_complex = np.zeros(result.shape[0], dtype=complex)
        contour_complex.real = result[:, 0]
        contour_complex.imag = result[:, 1]
        fd = np.fft.fft(contour_complex)
        fd_1dim = []
        for ele in fd:
            fd_1dim.append(ele.real)
        for ele in fd:
            fd_1dim.append(ele.imag)
        fd_1dim = np.array(fd_1dim)
        return fd_1dim


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
        return np.angle(np.exp(1j * angle))


class Angle2VecTransform(object):
    # def __init__(self):

    def __call__(self, angles):
        # convert radians to degree
        result = np.degrees(angles)
        # round degree to
        result = np.rint(result)
        # convert -180 degree to 180
        result[result == -180] = 180
        result += 179
        result = result.astype(int)
        vec = np.zeros((len(result), 360))
        for i in range(len(result)):
            vec[i, result[i]] = 1
        return vec


class RandomRotatePoints(object):
    '''
        single input means +/-, None means random, tuple defines range
    '''

    def __init__(self, rot_angle=None):
        assert isinstance(rot_angle, (int, float, type(None), tuple))
        self.rot_angle = rot_angle

    def __call__(self, points):
        rot_angle = self.rot_angle

        if isinstance(rot_angle, (int, float)):
            rot_angle_sample = np.random.uniform(low=-rot_angle, high=rot_angle)
            # print(rot_angle_sample)

        elif isinstance(rot_angle, tuple):
            rot_angle_sample = np.random.uniform(rot_angle[0], rot_angle[1])

        elif isinstance(rot_angle, type(None)):
            rot_angle_sample = np.random.uniform(0, 360)
            # print(rot_angle_sample)

        rot_angle_sample_rads = rot_angle_sample * (np.pi / 180.)

        rot_matrix = np.array([[np.cos(rot_angle_sample_rads), -np.sin(rot_angle_sample_rads)],
                               [np.sin(rot_angle_sample_rads), np.cos(rot_angle_sample_rads)]])
        points = rot_matrix.dot(points)
        # points = points.dot(rot_matrix)

        return index_start_from_left(points)


class RandomFlipPoints(object):

    def __init__(self, probability, vertical=False):
        assert isinstance(vertical, bool)
        assert isinstance(probability, (float, int))
        self.vertical = vertical
        self.probability = probability

    def __call__(self, points):
        vertical = self.vertical

        p = np.random.random()
        if p < self.probability:
            if vertical:
                if points.shape[0] == 2:
                    points[1, :] = -points[1, :]
                else:
                    points[:, 1] = -points[:, 1]
            else:
                if points.shape[0] == 2:
                    points[0, :] = -points[0, :]
                else:
                    points[:, 0] = -points[:, 0]

        return index_start_from_left(points)


class IndexRotate(object):

    def __call__(self, points):
        if points.shape[0] == 2:
            rand = random.randrange(points.shape[1])
            points = np.roll(points, rand, axis=1)
        else:
            rand = random.randrange(points.shape[0])
            points = np.roll(points, rand, axis=0)
        return points


class WhiteNoise(object):
    def __init__(self):
        pass

    def __call__(self, points):
        if points.shape[0] == 2:
            distance = np.random.normal(loc=0.0, scale=0.001, size=(2, len(points[0])))
            points += distance
        else:
            distance = np.random.normal(loc=0.0, scale=0.001, size=(len(points), 2))
            points += distance
        return points


class LowPassNoise(object):
    '''
        std_dev of the added gaussian noise is proportional to (abs(k)^-alpha) * beta
    '''

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, points):
        if points.shape[0] == 2:
            contour_complex = np.zeros(points.shape[1], dtype=complex)
            contour_complex.real = points[0, :]
            contour_complex.imag = points[1, :]
            fft_num = points.shape[1]
            fd = np.fft.fft(contour_complex, fft_num)
            fd.real = np.roll(fd.real, fft_num // 2)
            fd.imag = np.roll(fd.imag, fft_num // 2)
            noise = np.zeros(fd.shape, dtype=complex)
            k = np.array(list(reversed(range(1, fft_num // 2 + 1))) + [1] + list(range(1, fft_num // 2)))
            noise_real = [np.random.normal(0.0, self.beta * pow(i, self.alpha), size=1) for i in k]
            noise_imaginary = [np.random.normal(0.0, self.beta * pow(i, self.alpha), size=1) for i in k]
            noise.real = np.concatenate(noise_real)
            noise.imag = np.concatenate(noise_imaginary)
            noise.real[fft_num // 2] = 0
            noise.imag[fft_num // 2] = 0
            fd_noise = fd + noise
            fd_noise.real = np.roll(fd_noise.real, fft_num // 2)
            fd_noise.imag = np.roll(fd_noise.imag, fft_num // 2)
            ifd = np.fft.ifft(fd_noise)
            points = np.vstack([ifd.real, ifd.imag])
            return points


def index_start_from_left(points):
    if points.shape[1] == 2:
        idx = np.argmin(points[:, 0])
        return np.vstack((points[idx:], points[0:idx]))
    else:
        idx = np.argmin(points[0, :])
        return np.hstack((points[:, idx:], points[:, 0:idx]))


if __name__ == "__main__":
    # im = Image.open(r"D:\projects\shape_dataset\animal_dataset\duck\duck1.tif")
    # etf = EqualArclengthTransform(32)
    # ptf = PolygonTransform(32)
    # polygon = ptf(im)
    # result = etf(polygon)
    # print(result.shape)
    # plt.scatter(result[:, 0], result[:, 1])
    # plt.plot(result[:, 0], result[:, 1])
    # x = np.diff(result[:, 0])
    # y = np.diff(result[:, 1])
    # distance = np.square(x) + np.square(y)
    # plt.show()
    # print(np.square(x))
    # print(np.square(y))
    # print(distance)

    ##########Fourier Descriptor################
    # result = np.array([[-2, 2], [0, 0], [2, 2]])
    # contour_complex = np.zeros(result.shape[0], dtype=complex)
    # print(result[:, 0])
    # contour_complex.real = result[:, 0]
    # contour_complex.imag = result[:, 1]
    # fd = np.fft.fft(contour_complex)
    # print(fd)
    # fd_1dim = []
    # for ele in fd:
    #     fd_1dim.append(ele.real)
    # for ele in fd:
    #     fd_1dim.append(ele.imag)
    # fd_1dim = np.array(fd_1dim)
    # print(fd_1dim)
    ############### numpy #######################
    x = np.array(list(reversed(range(1, 16))) + [1] + list(range(1, 16)))
    print(x)
