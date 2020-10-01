import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.io import ImageCollection
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning, sparse_encode
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from scipy.signal import resample
from scipy.spatial.distance import cdist
from matplotlib.ticker import NullFormatter


def im2poly(image, n_samples, normalized=True, whiten=False):
    """Convert a shape image to a 2D polygon."""
    image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    # Extract contour coordinates
    # By default, `find_contours` returns contours oriented clockwise around
    # the higher values (shape), and the starting point is the minimum YX
    # coordinate in lexicographical order.
    coords = find_contours(image, level=0)[0]
    coords[:, 0], coords[:, 1] = coords[:, 1], -coords[:, 0] # lower left origin
    # Downsample the number of points by applying a low-pass filter (crop of
    # the Fourier transform)
    polygon = resample(coords, num=n_samples)
    if whiten:
        # Transform data to have identity covariance matrix
        polygon = PCA(n_components=2, whiten=True).fit_transform(polygon)
    if normalized:
        # Normalize to zero mean and unit norm
        polygon -= polygon.mean(axis=0)
        polygon /= np.linalg.norm(polygon, axis=0)[np.newaxis]
    return polygon


def occlude_shape(shape, n_points, start_point):
    """Occlude a shape (remove some consecutive points)."""
    ndim = shape.ndim
    if ndim == 1:
        # For 1-dimensional shapes (X then Y in a vector)
        xy = shape.shape[0] // 2
        shape = np.vstack((shape[:xy], shape[xy:])).T
    shape = np.roll(shape, -start_point)
    shape = shape[n_points:]
    if ndim == 1:
        shape = np.hstack((shape[:, 0], shape[:, 1]))
    return shape


def learn_sparse_components(shapes, n_components, lmbda, batch_size, n_iter=2000):
    """Learn sparse components from a dataset of shapes."""
    n_shapes = len(shapes)
    # Learn sparse components and predict coefficients for the dataset
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=lmbda,
                                     batch_size=batch_size, n_iter=n_iter, verbose=1)
    dl.coefficients = dl.fit_transform(shapes)
    # Compute frequency of activations and argsort
    # (but do not apply argsort as we would also need to sort coefficients and all inner
    # stats of the sklearn object)
    dl.frequencies = np.count_nonzero(dl.coefficients.T, axis=1) / n_shapes
    dl.argsort_freqs = np.argsort(-dl.frequencies)
    return dl


def learn_sparse_components2(shapes, n_components, lmbda, batch_size, transform_n_nonzero_coefs,n_iter=1000):
    """Learn sparse components from a dataset of shapes."""
    n_shapes = len(shapes)
    # Learn sparse components and predict coefficients for the dataset
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=lmbda,
                                     batch_size=batch_size, n_iter=n_iter, transform_n_nonzero_coefs=transform_n_nonzero_coefs, verbose=1)
    dl.coefficients = dl.fit_transform(shapes)
    # Compute frequency of activations and argsort
    # (but do not apply argsort as we would also need to sort coefficients and all inner
    # stats of the sklearn object)
    dl.frequencies = np.count_nonzero(dl.coefficients.T, axis=1) / n_shapes
    dl.argsort_freqs = np.argsort(-dl.frequencies)
    return dl


def learn_sparse_components3(shapes, n_components, lmbda, batch_size, transform_n_nonzero_coefs, fit_algorithm, n_iter=5000):
    """Learn sparse components from a dataset of shapes."""
    n_shapes = len(shapes)
    # Learn sparse components and predict coefficients for the dataset
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=lmbda,
                                     batch_size=batch_size, n_iter=n_iter, transform_n_nonzero_coefs=transform_n_nonzero_coefs, verbose=1, fit_algorithm=fit_algorithm, transform_algorithm='lasso_cd',
                                     positive_code=True)
    dl.coefficients = dl.fit_transform(shapes)
    # Compute frequency of activations and argsort
    # (but do not apply argsort as we would also need to sort coefficients and all inner
    # stats of the sklearn object)
    dl.frequencies = np.count_nonzero(dl.coefficients.T, axis=1) / n_shapes
    dl.argsort_freqs = np.argsort(-dl.frequencies)
    return dl


def learn_sparse_components4(shapes, n_components, lmbda, batch_size, fit_algorithm, n_iter=3000,transform_n_nonzero_coefs=None):
    """Learn sparse components from a dataset of shapes."""
    n_shapes = len(shapes)
    # Learn sparse components and predict coefficients for the dataset
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=lmbda,
                                     batch_size=batch_size, n_iter=n_iter, transform_n_nonzero_coefs=transform_n_nonzero_coefs, verbose=1, fit_algorithm=fit_algorithm)
    dl.coefficients = dl.fit_transform(shapes)
    # Compute frequency of activations and argsort
    # (but do not apply argsort as we would also need to sort coefficients and all inner
    # stats of the sklearn object)
    dl.frequencies = np.count_nonzero(dl.coefficients.T, axis=1) / n_shapes
    dl.argsort_freqs = np.argsort(-dl.frequencies)
    return dl

def plot_3d_components(components, figsize=None):
    from mpl_toolkits.mplot3d import Axes3D
    n_components, n_samples = components.shape
    xy = n_samples // 3
    yz = 2 * xy
    xx, yy, zz = components[:, :xy], components[:, xy:yz], components[:, yz:]

    xx = np.c_[xx, xx[:, 0]]
    yy = np.c_[yy, yy[:, 0]]
    zz = np.c_[zz, zz[:, 0]]

    xlim = np.min(xx), np.max(xx)
    ylim = np.min(yy), np.max(yy)
    zlim = np.min(zz), np.max(zz)

    fig = plt.figure(figsize=figsize if figsize is not None else (16, 16))
    rc = np.ceil(np.sqrt(n_components))
    for i in range(n_components):
        ax = plt.subplot(rc, rc, i+1, projection='3d')
        ax.plot(xx[i], yy[i], zz[i], c='C{}'.format(i % 10), linewidth=1.0)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        # ax.set_zlim(zlim)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.zaxis.set_major_formatter(NullFormatter())
    fig.tight_layout()
    return fig

def plot_components(phis, gains=None, domain='spatial', loop=True, nrows=None,
    ncols=None, figsize=None):
    """Plot shape components into a grid for different domains."""
    n_components, n_samples = phis.shape
    xy = n_samples // 2

    if domain not in ('spatial', 'fourier', '1D'):
        raise ValueError('Domain parameter must be `spatial`, `fourier` or `1D`.')

    if nrows and ncols and nrows * ncols < n_components:
        raise ValueError('`nrows` and `ncols` must match the number of components.')

    if gains is not None:
        if len(gains) != n_components:
            'Gains should be a vector of coefficients for the components.'
        phis *= gains[:, np.newaxis]
        # TODO: Maybe this shold not be applied here

    # Define global plot limits
    if domain == 'spatial':
        xlim = 1.1 * phis[:, :xy].min(), 1.1 * phis[:, :xy].max()
        ylim = 1.1 * phis[:, xy:].min(), 1.1 * phis[:, xy:].max()
    elif domain == 'fourier':
        complex_phis = phis[:, :xy] + 1j * phis[:, xy:]
        # phifts = np.fft.fftshift(np.fft.fft(complex_phis, axis=1), axes=1)
        # freqs = np.fft.fftshift(np.fft.fftfreq(xy, 1 / xy))
        phifts = np.fft.fft(complex_phis, axis=1)
        freqs = np.fft.fftfreq(xy, 1 / xy)[1:xy//2]
        # Take magnitude and sum over negative and positive frequencies
        pos_mags = np.abs(phifts[:, 1:(xy // 2)])
        neg_mags = np.abs(phifts[:, -((xy // 2) - 1):][:, ::-1]) # reverse array
        phifts = neg_mags + pos_mags
        ylim = (0, phifts.max() * 1.1)
    elif domain == '1D':
        ylim = (1.1 * phis.min(), 1.1 * phis.max())

    fig = plt.figure(figsize=figsize if figsize is not None else (8, 8))
    rc = np.ceil(np.sqrt(n_components))
    for i in range(n_components):
        if nrows and ncols:
            plt.subplot(nrows, ncols, i+1,)
        else:
            plt.subplot(rc, rc, i+1)
        if domain == 'spatial':
            plt.tick_params(axis='both', bottom=False, labelbottom=False,
                            left=False, labelleft=False,)
            phi = phis[i]
            if phi[0] < 0:
                phi = -phi
            plt.plot(phi[:xy], phi[xy:], c='C{}'.format(i%10), lw=1.5)
            if loop:
                plt.plot(np.array([phi[:xy][0], phi[:xy][-1]]), np.array([phi[xy:][0], phi[xy:][-1]]),
                         c='C{}'.format(i%10), lw=1.5)
            plt.xlim(xlim)
        elif domain == 'fourier':
            maxfreq = freqs[np.argmax(np.abs(phifts[i]))]
            plt.bar(freqs, phifts[i], width=0.8, alpha=0.75)
            plt.plot(freqs, phifts[i], 'o-')
            plt.xticks([0, 8, 16], [0, 8, 16])
            plt.xlabel('Frequency')
            plt.ylabel('Magnitude')
            plt.title('{}'.format((i*4)))
            # plt.yticks([])
            # plt.bar(freqs + 0.5, phifts[i].imag, width=0.5, align='edge')
            # plt.tick_params(axis='both', bottom=False, labelbottom=False,
                            # left=False, labelleft=False,)
            # plt.title('{}'.format(maxfreq))
            # plt.xlim((-n_samples/4, n_samples/4+1))
        elif domain == '1D':
            plt.tick_params(axis='both', bottom=False, labelbottom=False,
                            left=False, labelleft=False,)
            plt.plot(phis[i][:xy])
            plt.plot(phis[i][xy:])
        plt.ylim((ylim))

    return fig


def plot_reconstructions(shapes, dictionary, algorithm='omp',
        n_nonzero_coefs=None, alpha=None, show_points=False, show_errors=True,
        row_only=False, figsize=None):
    n_shapes = len(shapes)
    xy = shapes.shape[1] // 2
    coefficients = sparse_encode(shapes, dictionary, algorithm=algorithm,
                                 n_nonzero_coefs=n_nonzero_coefs, alpha=alpha)
    recons = np.dot(coefficients, dictionary)
    errors = np.sum((shapes - recons) ** 2, axis=1)

    if figsize is None:
        figsize = (8, 8)
    fig = plt.figure(figsize=figsize)
    markers = {'recons': '-o' if show_points else '-',
               'shapes': '--x' if show_points else '--'}
    rc = np.ceil(np.sqrt(n_shapes))

    for i in range(n_shapes):
        if row_only:
            plt.subplot(1, n_shapes, i+1)
        else:
            plt.subplot(rc, rc, i+1)
        plt.tick_params(axis='both', bottom=False, labelbottom=False,
                        left=False, labelleft=False,)
        plt.plot(shapes[i][:xy], shapes[i][xy:], markers['shapes'], c='C0', lw=1.0)
        plt.plot(recons[i][:xy], recons[i][xy:], markers['recons'], c='C1', lw=1.5)
        if show_errors:
            plt.title('Error: {:.2e}'.format(errors[i]))
    plt.tight_layout()

    return fig


def plot_reconstruction_detail(shape, dictionary, n_components, scaled=False,
        algorithm='omp', sorted=True, show_points=False, show_error=True,
        figsize=None):
    # Compute reconstruction for given shape
    xy = len(shape) // 2
    if algorithm in ('omp', 'lars'):
        coefs = sparse_encode(shape[np.newaxis, :], dictionary, algorithm=algorithm,
                              n_nonzero_coefs=n_components,)
        recons = np.dot(coefs, dictionary)[0]
    elif algorithm == 'pca':
        if not isinstance(dictionary, PCA):
            raise ValueError('Must pass PCA object for PCA algorithm.')
        pca = dictionary
        dictionary = pca.components_
        X = shape[np.newaxis, :] - pca.mean_
        coefs = np.dot(X, dictionary[:n_components].T)
        recons = np.dot(coefs, dictionary[:n_components])[0] + pca.mean_
    error = np.sum((shape - recons) ** 2)

    # Prepare plotting
    if figsize is None:
        figsize = ((5 * n_components), 8)
    fig = plt.figure(figsize=figsize)
    markers = {'recons': '-o' if show_points else '-',
               'shapes': '--x' if show_points else '--'}
    xlim = 1.1 * shape[:xy].min(), 1.1 * shape[:xy].max()
    ylim = 1.1 * shape[xy:].min(), 1.1 * shape[xy:].max()

    # # Plot the reconstruction along the initial shape
    # plt.subplot(2, n_components+1, 1)
    # plt.plot(shape[:xy], shape[xy:], markers['shapes'], c='C0', lw=1.0)
    # plt.plot(recons[:xy], recons[xy:], markers['recons'], c='C1', lw=1.5)
    # plt.xlim(xlim); plt.ylim(ylim);
    # plt.tick_params(axis='both', bottom=False, labelbottom=False,
    #                 left=False, labelleft=False,)
    # plt.title('Error = {:.4f}'.format(error), fontsize=18)

    # Plot the components sorted by coefficient values
    # as well as the cumulative sum
    argsort = np.argsort(-np.abs(coefs[0])) if sorted else np.arange(n_components)
    assert len(np.where(coefs != 0)[0]) == n_components
    cumsum = np.zeros_like(shape)
    if algorithm == 'pca':
        cumsum += pca.mean_
    for i in range(n_components):
        coef = coefs[0][argsort][i]
        comp = dictionary[argsort][i]
        prevsum = cumsum
        cumsum = cumsum + coef * comp
        error = np.sum((shape - cumsum) ** 2)
        if scaled:
            comp = coef * comp

        # if algorithm == 'pca':
        #     comp += pca.mean_
        # plt.subplot(2, n_components+1, 2+i)
        # plt.plot(comp[:xy], comp[xy:], markers['recons'],
        #          c='C{}'.format((i+2)%10), lw=1.5)
        # plt.xlim(xlim); plt.ylim(ylim)
        # plt.tick_params(axis='both', bottom=False, labelbottom=False,
        #                 left=False, labelleft=False,)
        # plt.title('{:.4f}'.format(coef), fontsize=18)
        plt.subplot(2, n_components, 1+i)
        plt.plot(comp[:xy], comp[xy:], markers['recons'],
                 c='C{}'.format((i+2)%10), lw=1.5)
        # loop
        plt.plot(np.array([comp[:xy][0], comp[:xy][-1]]),
                 np.array([comp[xy:][0], comp[xy:][-1]]),
                 markers['recons'], c='C{}'.format((i+2)%10), lw=1.5)
        plt.xlim(xlim); plt.ylim(ylim)
        plt.tick_params(axis='both', bottom=False, labelbottom=False,
                        left=False, labelleft=False,)
        plt.title('Coefficient = {:.2f}'.format(coef), fontsize=36)

        # plt.subplot(2, n_components+1, (n_components+3)+i)
        # plt.plot(prevsum[:xy], prevsum[xy:], markers['shapes'], lw=1.0)
        # plt.plot(cumsum[:xy], cumsum[xy:], markers['recons'], lw=1.5)
        # plt.xlim(xlim); plt.ylim(ylim)
        # plt.tick_params(axis='both', bottom=False, labelbottom=False,
        #                 left=False, labelleft=False,)
        plt.subplot(2, n_components, n_components+1+i)
        plt.plot(shape[:xy], shape[xy:], markers['shapes'], c='C0', lw=1.0)
        # loop
        plt.plot(np.array([shape[:xy][0], shape[:xy][-1]]),
                 np.array([shape[xy:][0], shape[xy:][-1]]),
                 markers['shapes'], c='C0', lw=1.0)
        plt.plot(cumsum[:xy], cumsum[xy:], markers['recons'], c='C1', lw=1.5)
        plt.plot(np.array([cumsum[:xy][0], cumsum[:xy][-1]]),
                 np.array([cumsum[xy:][0], cumsum[xy:][-1]]),
                 markers['recons'], c='C1', lw=1.5)
        plt.xlim(xlim); plt.ylim(ylim)
        plt.tick_params(axis='both', bottom=False, labelbottom=False,
                        left=False, labelleft=False,)
        plt.title('Error = {:.2f}'.format(error), fontsize=36)

    return fig


def plot_pca_harmonics(shape, pca, n_harmonics=6, figsize=None):
    xy = len(shape) // 2
    coefs = pca.transform(shape[np.newaxis, :])
    # recons = pca.inverse_transform(coefs)[0]
    recons = np.dot(coefs[0][:n_harmonics], pca.components_[:n_harmonics]) + pca.mean_
    error = np.sum((shape - recons) ** 2)
    # fund = fundamental_frequency(shape)

    # Prepare plotting
    if figsize is None:
        figsize = ((4 * n_harmonics) + 1, 7.5)
    fig = plt.figure(figsize=figsize)
    markers = {'recons': '-', 'shapes': '--'}
    xlim = 1.1 * shape[:xy].min(), 1.1 * shape[:xy].max()
    ylim = 1.1 * shape[xy:].min(), 1.1 * shape[xy:].max()

    # Plot the reconstruction along the initial shape
    plt.subplot(3, n_harmonics+2, 1)
    plt.plot(shape[:xy], shape[xy:], markers['shapes'], c='C0', lw=1.0)
    plt.plot(recons[:xy], recons[xy:], markers['recons'], c='C1', lw=1.5)
    plt.xlim(xlim); plt.ylim(ylim);
    plt.tick_params(axis='both', bottom=False, labelbottom=False,
                    left=False, labelleft=False,)

    # Plot the harmonics
    fund = pca.mean_
    plt.subplot(3, n_harmonics+2, 2)
    plt.plot(fund[:xy], fund[xy:], markers['recons'],
             c='C{}'.format((0+2)%10), lw=1.5)
    plt.title('Fundamental (PCA mean)')
    plt.xlim(xlim); plt.ylim(ylim)
    for i in range(n_harmonics):

        coef = coefs[0][i]
        comp = pca.components_[i]

        plt.subplot(3, n_harmonics+2, 3+i)
        plt.plot(comp[:xy], comp[xy:], markers['recons'],
                 c='C{}'.format((i+3)%10), lw=1.5)
        plt.xlim(xlim); plt.ylim(ylim)

        plt.subplot(3, n_harmonics+2, (n_harmonics+5)+i)
        harmonic = fund + comp
        plt.plot(fund[:xy], fund[xy:], markers['shapes'], lw=1.0)
        plt.plot(harmonic[:xy], harmonic[xy:], markers['recons'], lw=1.5)
        plt.xlim(xlim); plt.ylim(ylim)
        plt.title('No coefficient')

        plt.subplot(3, n_harmonics+2, (2*(n_harmonics+3)+1)+i)
        harmonic = fund + coef * comp
        plt.plot(fund[:xy], fund[xy:], markers['shapes'], lw=1.0)
        plt.plot(harmonic[:xy], harmonic[xy:], markers['recons'], lw=1.5)
        plt.xlim(xlim); plt.ylim(ylim)
        plt.title('Coefficient = {:.4f}'.format(coef))


    return fig


def sampling_residual_error(image, n_samples, normalized=True):
    image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    coords = find_contours(image, level=0)[0]
    coords[:, 0], coords[:, 1] = coords[:, 1], -coords[:, 0]
    n_points = coords.shape[0]
    polygon = resample(coords, num=n_samples)
    if normalized:
        polygon -= polygon.mean(axis=0)
        polygon /= np.linalg.norm(polygon, axis=0)[np.newaxis]
    # Generate same sampled polygon for different starting points
    polygons = [polygon,]
    step = n_points // 100 # try a hundred starting points
    for roll in range(step, n_points, step):
        shifted_coords = np.roll(coords, roll, axis=0)
        shifted_poly = resample(shifted_coords, n_samples)
        if normalized:
            shifted_poly -= shifted_poly.mean(axis=0)
            shifted_poly /= np.linalg.norm(shifted_poly, axis=0)[np.newaxis]
        # find closest to first starting point
        distances = [np.linalg.norm(polygon - np.roll(shifted_poly, i, axis=0)) for i in range(n_samples)]
        argmin = np.argmin(distances)
        shifted_poly = np.roll(shifted_poly, argmin, axis=0)
        polygons.append(shifted_poly)
    polygons = np.array(polygons)
    # Compute gramian matrix
    polygons_xy = np.hstack((polygons[:, :, 0], polygons[:, :, 0]))
    gramian = cdist(polygons_xy, polygons_xy)
    # Return mean distance between shifted polygons
    return (1 / n_samples) * np.mean(gramian[np.triu_indices(gramian.shape[0], 1)])


def turning_angles(shape):
    n_samples = shape.shape[0]
    # First compute difference between consecutive points (assuming closed curves)
    closed_shape = np.vstack((shape, shape[0, np.newaxis]))
    differences = np.gradient(closed_shape, axis=0)
    # Then compute angle between these consecutive differences
    angles = np.zeros(n_samples)
    indices = np.arange(-1, n_samples); indices[-1] = 0 # start with -1 and finish with 0
    for k, i in enumerate(indices[:-1]):
        a, b = differences[i], differences[i+1]
        anorm, bnorm = np.linalg.norm(a), np.linalg.norm(b)
        # arccos restricts the angle to [0, pi]
        angles[k] = np.arccos(np.dot(a, b) / (anorm * bnorm))
    return angles


def curvature(shape):
    # Use finite differences method in Numpy to compute derivatives
    dx, dy = np.gradient(shape[:, 0]), np.gradient(shape[:, 1])
    d2x, d2y = np.gradient(dx), np.gradient(dy)
    return np.abs(d2x * dy - d2y * dx) / (dx ** 2 + dy ** 2) ** 1.5


def fundamental_frequency(shape):
    """Return fundamental ellipse for a given shape."""
    # Shape is expected to be in complex number format
    n_samples = len(shape)
    # Apply Fourier transform to retrieve frequencies
    ft = np.fft.fft(shape)
    fund_pos, fund_neg = ft[1], ft[-1]
    # Get ellipse back from fundamentals
    iexp = np.exp(2j * np.pi * np.arange(n_samples) / n_samples)
    # for fund_neg, exp(-x) = 1 / exp(x)
    fund = (1 / n_samples) * (fund_pos * iexp + fund_neg * iexp ** -1)
    return fund


def normalize_orientations(shapes):
    # Apply PCA transform to each shape individually
    rshapes = np.zeros_like(shapes)
    for i, shape in enumerate(shapes):
        shape = np.stack((shape[:32], shape[32:])).T
        rshape = PCA().fit_transform(shape)
        rshapes[i] = np.hstack((rshape[:, 0], rshape[:, 1]))
    # Shapes are now aligned with the fundamental ellipse
    # To be less sensible to starting point, shift to maximum x coordinate
    # corresponding to the axis of maximum variance
    argmax = np.argmax(rshapes[:, :32], axis=1)
    shapes_shifted = np.zeros_like(rshapes)
    for i, shape in enumerate(rshapes):
        x = np.roll(shape[:32], -argmax[i])
        y = np.roll(shape[32:], -argmax[i])
        shapes_shifted[i] = np.hstack((x, y))
    return shapes_shifted


def log_frequency_coefficients(components):
    n_components, n_samples = components.shape
    xy = n_samples // 2
    # Fourier transform, ignoring the DC component
    ft = np.fft.fft(components[:, :xy] + 1j * components[:, xy:])[:, 1:]
    freqs = np.fft.fftfreq(xy, 1 / xy)[1:]
    # Take the log
    log_ft = np.log(np.abs(ft))
    log_freqs = np.log(np.abs(freqs))
    # Find best fitting line for each component
    alphas, intercepts = np.zeros(n_components), np.zeros(n_components)
    for i in range(n_components):
        lr = LinearRegression()
        lr.fit(log_freqs[:, np.newaxis], log_ft[i][:, np.newaxis])
        alphas[i], intercepts[i] = lr.coef_[0, 0], lr.intercept_
    return alphas, intercepts


def max_frequency_magnitudes(components):
    n_components, n_samples = components.shape
    xy = n_samples // 2
    # Fourier transform
    ft = np.fft.fft(components[:, :xy] + 1j * components[:, xy:])
    # Take magnitude and sum over negative and positive frequencies
    pos_mags = np.abs(ft[:, 1:(xy // 2)])
    neg_mags = np.abs(ft[:, -((xy // 2) - 1):][:, ::-1]) # reverse array
    magnitudes = neg_mags + pos_mags
    return magnitudes
