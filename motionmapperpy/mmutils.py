import copy
import os
import logging

import h5py
import hdf5storage
import matplotlib as mpl
import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.io import loadmat
logger=logging.getLogger(__name__)

try:
    from awkde import GaussianKDE
except:
    logger.warning("awkde not found")
    GaussianKDE=None


def gencmap():
    """
    Get behavioral map colormap as a matplotlib colormap instance.
    :return: Matplotlib colormap instance.
    """
    colors = np.zeros((64, 3))
    colors[:21, 0] = np.linspace(1, 0, 21)
    colors[20:43, 0] = np.linspace(0, 1, 23)
    colors[42:, 0] = 1.0

    colors[:21, 1] = np.linspace(1, 0, 21)
    colors[20:43, 1] = np.linspace(0, 1, 23)
    colors[42:, 1] = np.linspace(1, 0, 22)

    colors[:21, 2] = 1.0
    colors[20:43, 2] = np.linspace(1, 0, 23)
    colors[42:, 2] = 0.0
    return mpl.colors.ListedColormap(colors)


def createProjectDirectory(pathToProject):
    _dirs = [
        pathToProject,
        "%s/Projections" % pathToProject,
        "%s/TSNE_Projections" % pathToProject,
        "%s/TSNE" % pathToProject,
        "%s/UMAP" % pathToProject,
        "%s/Wavelets" % pathToProject,
        "%s/Subsampled_wavelets" % pathToProject,
        "%s/Ego" % pathToProject,
        "%s/Processed_h5s" % pathToProject,
    ]
    for d in _dirs:
        if not os.path.exists(d):
            print("Creating : %s" % d)
            os.mkdir(d)
        else:
            print("Skipping, path already exists : %s" % d)
    return


def getDensityBounds(density, thresh=1e-6):
    """
    Get the outline for density maps.
    :param density: m by n density image.
    :param thresh: Density threshold for boundaries. Default 1e-6.
    :return: (p by 2) points outlining density map.
    """
    x_w, y_w = np.where(density > thresh)
    x, inv_inds = np.unique(x_w, return_inverse=True)
    bounds = np.zeros((x.shape[0] * 2 + 1, 2))
    for i in range(x.shape[0]):
        bounds[i, 0] = x[i]
        bounds[i, 1] = np.min(y_w[x_w == bounds[i, 0]])
        bounds[x.shape[0] + i, 0] = x[-i - 1]
        bounds[x.shape[0] + i, 1] = np.max(y_w[x_w == bounds[x.shape[0] + i, 0]])
    bounds[-1] = bounds[0]
    bounds[:, [0, 1]] = bounds[:, [1, 0]]
    return bounds.astype(int)


def findPointDensity_awkde(zValues, alpha, glob_bw, numPoints, rangeVals):
    """
    findPointDensity finds a Kernel-estimated PDF from a set of 2D data points
    through convolving with a Gaussian function.
    :param zValues: 2d points of shape (m by 2).
    :param alpha: A smoothing factor for Gaussian KDE.
    :param glob_bw: A bandwidth factor for Gaussian KDE.
    :param numPoints: Output density map dimension (n x n).
    :param rangeVals: 1 x 2 array giving the extrema of the observed range
    :return:
        bounds -> Outline of the density map (k x 2).
        xx -> 1 x numPoints array giving the x and y axis evaluation points.
        density -> numPoints x numPoints array giving the PDF values (n by n) density map.
    """
    # zValues = zValues[np.random.choice(zValues.shape[0], 1000, replace=False), :]
    xx = np.linspace(rangeVals[0], rangeVals[1], numPoints)
    yy = copy.copy(xx)

    # Use the same meshgrid as in the other function
    [XX, YY] = np.meshgrid(xx, yy)
    grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T

    density = compute_density_map(zValues, alpha, glob_bw, grid_pts, numPoints)

    # Normalize density so it sums to 1
    density = density / np.sum(density)

    # Find bounds as in the other function
    bounds = getDensityBounds(density)
    return bounds, xx, density


def compute_density_map(zValues, alpha, glob_bw, grid_pts, numPoints):
    if GaussianKDE is None:
        raise ModuleNotFoundError("awkde not installed")
    kde = GaussianKDE(glob_bw=glob_bw, alpha=alpha, diag_cov=True)
    kde.fit(zValues)

    zz = kde.predict(grid_pts)
    ZZ = zz.reshape(numPoints, numPoints)  # Reshape back to grid size

    return ZZ  # Return the density map for further processing


def findPointDensity(zValues, sigma, numPoints, rangeVals):
    """
        findPointDensity finds a Kernel-estimated PDF from a set of 2D data points
        through convolving with a gaussian function.
        :param zValues: 2d points of shape (m by 2).
        :param sigma: standard deviation of smoothing gaussian.
        :param numPoints: Output density map dimension (n x n).
        :param rangeVals: 1 x 2 array giving the extrema of the observed range
        :return:
            bounds -> Outline of the density map (k x 2).
            xx -> 1 x numPoints array giving the x and y axis evaluation points.
    %       density -> numPoints x numPoints array giving the PDF values (n by n) density map.
    """
    xx = np.linspace(rangeVals[0], rangeVals[1], numPoints)
    yy = copy.copy(xx)
    # TODO
    # Use awkde to get density instead...
    [XX, YY] = np.meshgrid(xx, yy)
    G = np.exp(-0.5 * (np.square(XX) + np.square(YY)) / np.square(sigma))
    Z = np.histogramdd(zValues, bins=[xx, yy])[0]
    Z = Z / np.sum(Z)
    Z = np.pad(Z, ((0, 1), (0, 1)), mode="constant", constant_values=((0, 0), (0, 0)))
    density = fftshift(np.real(ifft2(np.multiply(fft2(G), fft2(Z))))).T
    # density[density < 1e-4] = 0
    # density[density > 1.5e-2] = 1.5e-2
    bounds = getDensityBounds(density)
    return bounds, xx, density


def randomizewshed(wshed):
    outwshed = np.zeros_like(wshed)
    regs = np.unique(wshed)[1:]
    np.random.shuffle(regs)
    for i, wreg in enumerate(regs):
        outwshed[wshed == wreg] = i
    return outwshed


def getPDF(x, mu, sigma, p):
    return (p / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(
        -0.5 * (x - mu) ** 2 / sigma**2
    )


def conV2matV7(matfile):
    try:
        a = loadmat(matfile)
        _ = [a.pop(i) for i in list(a.keys()) if "__" in i]
        hdf5storage.write(
            data=a,
            path="/",
            truncate_existing=True,
            filename=matfile,
            store_python_metadata=False,
            matlab_compatible=True,
        )
        print("File converted to MATLAB -v7.3")
        return
    except Exception as E:
        try:
            h5file = h5py.File(matfile, "r")
            print("File already to MATLAB -v7.3")
            return
        except Exception as F:
            print("Something bad happened. File could be lost.")
            print(E)
            print(F)
            return


def checkParams(parameters, projections):
    if np.any([p.shape[0] < parameters.numPoints for p in projections]):
        plens = [p.shape[0] for p in projections]
        print(plens)
        print(
            "Training number of points for miniTSNE is greater than # samples in some files. Adjust it to "
            "smallest # samples : %i" % (np.min(plens))
        )
