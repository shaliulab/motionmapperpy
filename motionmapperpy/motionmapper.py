import itertools
import logging
import glob
import multiprocessing as mp
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")

import pickle
import logging
from pathlib import Path

import h5py
import codetiming
import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from scipy.io import loadmat, savemat
from scipy.optimize import fmin
from scipy.spatial import Delaunay, distance
from skimage.filters import roberts
from skimage.segmentation import watershed
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from umap import UMAP as cpuUMAP
import joblib
import numpy as np
from scipy.signal import lombscargle

from .mmutils import findPointDensity, gencmap
from .setrunparameters import setRunParameters
from .wavelet import findWavelets

# MIN_POWER=4e2
MIN_POWER=1

logger = logging.getLogger(__name__)

try:
    from cuml.manifold.umap import UMAP as cumlUMAP
except ModuleNotFoundError:
    cumlUMAP=None


"""Core t-SNE MotionMapper functions."""


def findKLDivergences(data):
    N = len(data)
    logData = np.log(data)
    logData[~np.isfinite(logData)] = 0

    entropies = -np.sum(np.multiply(data, logData), 1)

    D = -np.dot(data, logData.T)

    D = D - entropies[:, None]

    D = D / np.log(2)
    np.fill_diagonal(D, 0)
    return D, entropies


def run_UMAP(data, parameters, save_model=True, metric="euclidean"):
    """
    Project data to lower dimensionality space computed using the UMAP algorithm

    Arguments:

        data (np.array): Dataset with observations in the rows and features in the columns
    
    Returns:
        y (np.array): Mean centered projected dataset
    """
    if not parameters.waveletDecomp:
        raise ValueError("UMAP not implemented without wavelet decomposition.")
    print("Running UMAP with metric: " + parameters.umapMetric)
    # TODO: Determine if want this normalization
    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    umapfolder = parameters["projectPath"] + "/UMAP/"
    (
        n_neighbors,
        train_negative_sample_rate,
        min_dist,
        umap_output_dims,
        n_training_epochs,
    ) = (
        parameters["n_neighbors"],
        parameters["train_negative_sample_rate"],
        parameters["min_dist"],
        parameters["umap_output_dims"],
        parameters["n_training_epochs"],
    )

    if parameters.useGPU >= 0:
        assert cumlUMAP is not None, f"cuml is not available. Please set useGPU to -1"
        UMAP=cumlUMAP
        n_training_epochs=2000
        print("Using GPU UMAP")
    else:
        UMAP=cpuUMAP
        print("Using CPU UMAP")
    
    um = UMAP(
        n_neighbors=n_neighbors,
        negative_sample_rate=train_negative_sample_rate,
        min_dist=min_dist,
        n_components=umap_output_dims,
        n_epochs=n_training_epochs,
        metric=metric,  # TODO: check if this is the right metric
    )

    y = um.fit_transform(data)
    trainmean = np.mean(y, 0)
    scale = parameters["rescale_max"] / np.abs(y).max()
    y = y - trainmean
    y = y * scale

    if save_model:
        print("Saving UMAP model to disk...")
        np.save(
            umapfolder + "_trainMeanScale.npy",
            np.array([trainmean, scale], dtype=object),
        )
        if parameters.useGPU>=0:

            model_file=umapfolder+"umap.joblib"
            joblib.dump(um, model_file)

        else:
            model_file=umapfolder + "umap.model"
            with open(model_file, "wb") as f:
                pickle.dump(um, f)

    return y


def run_tSne(data, parameters=None, filename="none"):
    """
    run_tSne runs the t-SNE algorithm on an array of normalized wavelet amplitudes
    :param data: Nxd array of wavelet amplitudes (will normalize if unnormalized) containing N data points
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
            yData -> N x 2 array of embedding results
    """
    parameters = setRunParameters(parameters)

    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    if parameters.waveletDecomp:
        print("Finding Distances")
        D, _ = findKLDivergences(data)
        D[~np.isfinite(D)] = 0.0
        D = np.square(D)
        dist_mat_mean = np.mean(D)
        print(f"Distance matrix shape: {D.shape}")
        print(f"Distance matrix mean: {dist_mat_mean}")
        if dist_mat_mean < 0.00001:
            print("Distance matrix mean is too small. Adding to bad file list.")
            with open("list_of_bad_files.txt", "a") as txt_file:
                txt_file.write(f"{filename} \n")

        print("Computing t-SNE with %s method" % parameters.tSNE_method)
        tsne = TSNE(
            perplexity=parameters.perplexity,
            metric="precomputed",
            verbose=1,
            n_jobs=-1,
            method=parameters.tSNE_method,
        )
        yData = tsne.fit_transform(D)
    else:
        print("TSNE fitting complete. Computing Distances")
        tsne = TSNE(
            perplexity=parameters.perplexity,
            learning_rate="auto",
            metric="euclidean",
            verbose=1,
            n_jobs=-1,
            method=parameters.tSNE_method,
            n_iter=parameters.maxOptimIter,
        )
        yData = tsne.fit_transform(data)
        # raise ValueError('tSNE not implemented for runs without wavelet decomposition.')
    return yData


"""Training-set Generation"""


def returnTemplates(yData, signalData, minTemplateLength=10, kdNeighbors=10):
    maxY = np.ceil(np.max(np.abs(yData[:]))) + 1

    nn = NearestNeighbors(n_neighbors=kdNeighbors + 1, n_jobs=-1)
    nn.fit(yData)
    D, _ = nn.kneighbors(yData)
    sigma = np.median(D[:, -1])

    _, xx, density = findPointDensity(yData, sigma, 501, [-maxY, maxY])

    L = watershed(-density, connectivity=10)

    watershedValues = np.digitize(yData, xx)
    watershedValues = L[watershedValues[:, 1], watershedValues[:, 0]]

    maxL = np.max(L)

    templates = []
    for i in range(1, maxL + 1):
        templates.append(signalData[watershedValues == i])
    lengths = np.array([len(i) for i in templates])
    templates = np.array(templates, dtype=object)

    idx = np.where(lengths >= minTemplateLength)[0]
    vals2 = np.zeros(watershedValues.shape)
    for i in range(len(idx)):
        vals2[watershedValues == idx[i] + 1] = i + 1

    templates = templates[lengths >= minTemplateLength]
    lengths = lengths[lengths >= minTemplateLength]

    return templates, xx, density, sigma, lengths, L, vals2


def findTemplatesFromData(
    signalData, yData, signalAmps, numPerDataSet, parameters, projectionFile
):
    kdNeighbors = parameters.kdNeighbors
    minTemplateLength = parameters.minTemplateLength

    print("Finding Templates.")
    templates, _, density, _, templateLengths, L, vals = returnTemplates(
        yData, signalData, minTemplateLength, kdNeighbors
    )

    ####################################################
    wbounds = np.where(roberts(L).astype("bool"))
    wbounds = (wbounds[1], wbounds[0])
    fig, ax = plt.subplots()
    ax.imshow(density, origin="lower", cmap=gencmap())
    ax.scatter(wbounds[0], wbounds[1], color="k", s=0.1)
    fig.savefig(projectionFile[:-4] + "_trainingtSNE.png")
    plt.close()
    print(f"Saved training tSNE plot to {projectionFile[:-4]+'_trainingtSNE.png'}")
    ####################################################

    N = len(templates)
    d = len(signalData[1, :])
    selectedData = np.zeros((numPerDataSet, d))
    selectedAmps = np.zeros((numPerDataSet, 1))

    numInGroup = np.round(numPerDataSet * templateLengths / np.sum(templateLengths))
    numInGroup[numInGroup == 0] = 1
    sumVal = np.sum(numInGroup)
    if sumVal < numPerDataSet:
        q = int(numPerDataSet - sumVal)
        idx = np.random.permutation(N)[: min(q, N)]
        numInGroup[idx] = numInGroup[idx] + 1
    else:
        if sumVal > numPerDataSet:
            q = int(sumVal - numPerDataSet)
            idx2 = np.where(numInGroup > 1)[0]
            Lq = len(idx2)
            if Lq < q:
                idx2 = np.arange(len(numInGroup))
            idx = np.random.permutation(len(idx2))[:q]
            numInGroup[idx2[idx]] = numInGroup[idx2[idx]] - 1
    idx = numInGroup > templateLengths
    numInGroup[idx] = templateLengths[idx]
    cumSumGroupVals = [0] + np.cumsum(numInGroup).astype(int).tolist()

    for j in range(N):
        if cumSumGroupVals[j + 1] > cumSumGroupVals[j]:
            amps = signalAmps[vals == j + 1]
            idx2 = np.random.permutation(len(templates[j][:, 1]))[
                : int(numInGroup[j])
            ].astype(int)
            selectedData[cumSumGroupVals[j] : cumSumGroupVals[j + 1], :] = templates[j][
                idx2, :
            ]
            selectedAmps[cumSumGroupVals[j] : cumSumGroupVals[j + 1], 0] = amps[idx2]

    signalData = selectedData
    signalAmps = selectedAmps

    return signalData, signalAmps


def mm_findWavelets(projections, numModes, parameters):
    amplitudes, f = findWavelets(
        projections,
        numModes,
        parameters.omega0,
        parameters.numPeriods,
        parameters.samplingFreq,
        parameters.maxF,
        parameters.minF,
        parameters.numProcessors,
        parameters.useGPU,
    )
    return amplitudes, f


import pathlib


def file_embeddingSubSampling(projectionFile, parameters):
    perplexity = parameters.training_perplexity

    if parameters.waveletDecomp:
        print("\n Loading wavelets")
        # projections = np.array(loadmat(projectionFile, variable_names=['projections'])['projections'])
        with h5py.File(
            f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat",
            "r",
        ) as f:
            data = f["signaldata"][:]  # [signalIdx]
            # subsampled timepoints x (freqs * n_pcs)

        print(f"Data shape: {data.shape}")
        print("\n Loaded wavelets")
        # data = loadmat(f'{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat')

        print("\n Subsampled wavelets")

        signalData = data

    print("\n Subsampled projections")
    signalAmps = np.sum(signalData, axis=1)

    signalData = signalData / signalAmps[:, None]

    if parameters.method == "TSNE":
        parameters.perplexity = perplexity
        yData = run_tSne(signalData, parameters, projectionFile)
    elif parameters.method == "UMAP":
        yData = run_UMAP(
            signalData,
            parameters,
            save_model=False,
            metric=parameters.umapSubsampMetric,
        )
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")
    return yData, signalData, np.arange(parameters.training_numPoints), signalAmps


from tqdm import tqdm


def get_wavelets(projectionFiles, parameters, i, ls=False, cache=None):

    if isinstance(i, int):
        projectionFile=projectionFiles[i]
    elif isinstance(i, str):
        projectionFiles=[projectionFile for projectionFile in projectionFiles if os.path.splitext(os.path.basename(projectionFile))[0]==i]
        if len(projectionFiles)>1 or len(projectionFiles) == 0:
            raise ValueError(f"{len(projectionFiles)} match with {i}")
        projectionFile=projectionFiles[0]
    else:
        raise ValueError("Please pass an integer or a dataset name")

    print(f"Processing {projectionFile}")
    if ls:
        calc_and_write_wavelets_ls(projectionFile, parameters, cache=cache)
    else:
        calc_and_write_wavelets(projectionFile, parameters)


def mm_findWavelets_ls(projections, parameters):
    """

    Args:
        projections (np.array): Pose stored as a table where rows represent frames and columns represent features
        parameters (OrderedDict): Obtained with mmpy.setRunParameters()

    Returns:
        amplitudes.T (np.array): frames x frequencies. All frequencies of the first body part come first, then for the second body part, and so on
        f (list): frequencies screened
        window_sizes: the ith element contains the amount of data (number of contiguous time points) fed to the algorithm to compute the power of the ith frequency
    """

    t1 = time.time()
    logger.info("\t Calculating wavelets, clock starting.")

    import multiprocessing as mp
    import numpy as np

    if parameters.numProcessors < 0:
        parameters.numProcessors = mp.cpu_count()
    logger.info("\t Using #%i CPUs." % parameters.numProcessors)
    logger.info("Using Lomb-Scargle.")

    projections = np.array(projections)

    t1 = time.time()

    minT = 1.0 / parameters.maxF
    maxT = 1.0 / parameters.minF
    Ts = minT * (
        2
        ** (
            (np.arange(parameters.numPeriods) * np.log(maxT / minT))
            / (np.log(2) * (parameters.numPeriods - 1))
        )
    )
    f = (1.0 / Ts)[::-1]

    # TODO: Move this to parameters
    # This expression computes the needed window sizes for each frequency
    omega0 = parameters.omega0
    scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * f)

    window_sizes = np.round(scales * parameters.samplingFreq).astype(int)
    logger.info(f"Window sizes: {window_sizes}")
    logger.info(f"Frequencies: {f}, shape: {f.shape}")

    N = projections.shape[0]
    logger.info(f"Projection shape: {projections.shape}")
    logger.info("No normalization -- precentering though.")
    indices=np.arange(0, projections.shape[0], parameters.wavelet_downsample)
    
    with codetiming.Timer(text="Lomb-Scargle algorithm: {:.2f} seconds"):
        try:
            if parameters.numProcessors>1:
                pool = mp.Pool(parameters.numProcessors)
                logger.info(f"Scarglin' {projections.shape[1]} projections")
                amplitudes = pool.starmap(
                    rolling_lombscargle,
                    [
                        (
                            projections[:, i],
                            np.linspace(0, N / parameters.samplingFreq, N),
                            f.astype(float),
                            window_sizes,
                            parameters.wavelet_downsample
                        )
                        for i in range(projections.shape[1])
                    ],
                )
            else:
                amplitudes=[]
                for i in range(projections.shape[1]):
                    amplitudes.append(
                        rolling_lombscargle(
                            projections[:, i],
                            np.linspace(0, N / parameters.samplingFreq, N),
                            f.astype(float),
                            window_sizes,
                            skip=parameters.wavelet_downsample
                        )
                    )
                    
            amplitudes = np.concatenate(amplitudes, 0)
            amplitudes[~np.isfinite(amplitudes)] = 0
            logger.info(f"Done Scarglin' -- shape: {amplitudes.shape}")
            pool.close()
            pool.join()
        except Exception as E:
            pool.close()
            pool.join()
            raise E

    logger.info("\t Done at %0.02f seconds." % (time.time() - t1))
    return amplitudes.T, f, window_sizes, indices


def rolling_window_with_padding(arr, window_size, skip=1):
    """
    Create a set of rolling windows based on some input and a window size

    Windows are created from the input so that as many windows as positions in the input are available
    by padding the edges with the first and last values of the input (edge mode)
    Once the input is edge-padded, the windows are themselves created using a numpy trick
    to not copy the same values in memory

    Returns:
        If arr is 1D:
            the output has the same number of rows as the length of the input
            each row represents one window
            the number of columns is the window size
        
        If arr is > 1D:
            ?
    """

    # TODO: double check this
    padding = (window_size - 1) // 2
    padded_arr = np.pad(arr, (padding, padding), mode="edge")
    shape = padded_arr.shape[:-1] + (
        padded_arr.shape[-1] - window_size + 1,
        window_size,
    )
    # assert shape[0] == padded_arr.shape[0], f"{shape[0]} != {padded_arr.shape[0]}"
    strides = padded_arr.strides + (padded_arr.strides[-1],)
    strides=tuple([strides[0]*skip] + list(strides[1:]))
    shape=tuple([shape[0]//skip] + list(shape[1:]))


    return np.lib.stride_tricks.as_strided(padded_arr, shape=shape, strides=strides)


def rolling_lombscargle(data, sampling_times, freqs, window_sizes, skip=1):
    """
    Compute the lombgscargle periodogram of a timeseries

    The power of each frequency provided in `freqs` is computed at every position of the 
    timeseries using the data provided within a window of a corresponding window_size

    The window size of the ith frequency is given by the ith position in the window_sizes array
    The data is not required to be equidistnat over time, and instead you need to pass the sampling point
    t_i of each datapoint d_i

    Args:

        data (np.array): 1D timeseries
        sampling_times (np.array): The ith position contains the sampling time of the ith data point
        freqs (np.array): Frequencies for which the power should be computed 
        window_sizes (np.array): Amount of time used to compute the frequencies. Low frequencies require less time
            because the signal changes more slowly, which means less temporal resolution is possible. Therefore,
            you should pass window sizes that increase in size as the frequencies decrease
            
    
    Returns:
        periodograms.T (np.array): frequencies x windows

    Learn more -> https://youtu.be/y7KLbd7n75g?si=o8KideRnzhBYpFKx

    """

    # Initialize an empty array to store the Lomb-Scargle periodograms
    periodograms = np.zeros((data.size, freqs.size))

    # Loop through each frequency and its corresponding window size
    for f_idx, (freq, window_size) in enumerate(zip(freqs, window_sizes)):
        logger.debug(f"On frequency {f_idx} of {freqs.size}")
        # print(
        #     f"Inside rolling_lombscargle -  freq: {freq}, win size: {window_size}"
        # )  # Debug print
        # print(f"Inside rolling_lombscargle -  freq: {freq}")  # Debug print
        windows = rolling_window_with_padding(data, window_size, skip=1)
        number_of_windows, window_size_ = windows.shape

        # print(
        #     f"Inside rolling_lombscargle -  windows shape: {windows.shape}"
        # )  # Debug print
        windows_sampling_times = rolling_window_with_padding(
            sampling_times, window_size, skip=1
        )

        indices=np.arange(0, number_of_windows, skip)
        windows=windows[::skip, ...]
        number_of_windows, window_size_ = windows.shape
        windows_sampling_times=windows_sampling_times[::skip, ...]
        assert len(indices) == number_of_windows


        for i, (window, times) in enumerate(zip(windows, windows_sampling_times)):
            angular_frequency = 2 * np.pi * freq
            tmp_window = window.copy()
            # print(f"Inside rolling_lombscargle -  window shape: {window.shape}")
            window = window[np.isfinite(tmp_window)]
            # print(f"Post nan removal -  window shape: {window.shape}")
            # sampling_times = sampling_times[np.isfinite(tmp_window)]
            sampling_times_window = times[np.isfinite(tmp_window)]
            # print(f"Processing window {i} of {windows.shape[0]}")
            periodogram = lombscargle(
                sampling_times_window,
                window,
                [angular_frequency],
                normalize=False,
                precenter=True,
            )

            if np.all(np.isnan(periodogram)):
                periodogram = 0

            periodograms[indices[i], f_idx] = periodogram
    periodograms=periodograms[indices, :]
    return periodograms.T


def calc_and_write_wavelets_ls(projectionFile, parameters, cache=None):
    # calculate and write wavelets with lomb-scargle from scipy
    print("\t Loading Projections")

    with h5py.File(projectionFile, "r") as hfile:
        projections = hfile["projections"][:].T
        node_names = hfile["node_names"][:]
    projections = np.array(projections)
    if cache is not None:
        final_output=f"{cache}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        cache_exists=os.path.exists(final_output)
    else:
        cache_exists=False

    if parameters.waveletDecomp:
        output_file=f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"

        if not os.path.exists(
            output_file
        ):
            
            if cache_exists:
                print(f"\t Restoring cached {final_output}. {output_file} becomes a link to it")
                #shutil.copy(final_output, f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat")
                os.symlink(final_output, output_file)
            else:
                print("\t Calculating Wavelets")
                data, freqs, win_sizes, indices = mm_findWavelets_ls(projections, parameters)
                freq_names=[]
                for part in node_names:
                    for freq in freqs:
                        freq_names.append(
                            part.decode() + "_x_" + str(round(freq, 4))
                        )
                        freq_names.append(
                            part.decode() + "_y_" + str(round(freq, 4))
                        )
                
                assert len(freq_names) == data.shape[1]
                print(f"\n Saving wavelets: {data.shape}")
                with h5py.File(
                    output_file,
                    "w",
                    libver="latest",
                ) as f:
                    print("No compression")
                    f.create_dataset("wavelets", data=data)
                    f.create_dataset("indices", data=indices)
                    f.create_dataset("f", data=freqs)
                    f.create_dataset("win_sizes", data=win_sizes)
                    f.create_dataset("node_names", data=node_names)
                    f.create_dataset("freq_names", data=freq_names)


def calc_and_write_wavelets(projectionFile, parameters):
    print("\t Loading Projections")

    with h5py.File(projectionFile, "r") as hfile:
        projections = hfile["projections"][:].T
    projections = np.array(projections)

    if parameters.waveletDecomp:
        print("\t Calculating Wavelets")
        if not os.path.exists(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        ):
            data, freqs = mm_findWavelets(projections, parameters.pcaModes, parameters)
            print(f"\n Saving wavelets: {data.shape}")
            with h5py.File(
                f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
                "w",
                libver="latest",
            ) as f:
                print("No compression")
                f.create_dataset("wavelets", data=data)
                f.create_dataset("f", data=freqs)


import natsort

from tqdm import tqdm


def runEmbeddingSubSampling(projectionDirectory, parameters):
    """
    runEmbeddingSubSampling generates a training set given a set of .mat files.

    :param projectionDirectory: directory path containing .mat projection files.
    Each of these files should contain an N x pcaModes variable, 'projections'.
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
        trainingSetData -> normalized wavelet training set
                           (N x (pcaModes*numPeriods) )
        trainingSetAmps -> Nx1 array of training set wavelet amplitudes
        projectionFiles -> list of files in 'projectionDirectory'
    """
    parameters = setRunParameters(parameters)
    projectionFiles = glob.glob(projectionDirectory + "/*pcaModes.mat")
    projectionFiles = natsort.natsorted(projectionFiles)
    for projectionFile in projectionFiles.copy():
        print(f"Checking {projectionFile}")
        if not os.path.exists(
            f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat"
        ):
            print(f"Skipping {projectionFile}")
            projectionFiles.remove(projectionFile)
    n_requested = parameters.trainingSetSize
    L = len(projectionFiles)
    if L == 0:
        raise Exception("No subsampled wavelets found")

    n_requested_per_dataset = round(n_requested / L)
    print(f"Number of files: {L}")
    print(f"Number of samples per file: {n_requested_per_dataset}")
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods

    # this happens if the requested per dataset is more than
    # the size of the subsampled datasets, because either the former is too small
    # or because we are asking for too many training points
    if n_requested_per_dataset > parameters.training_numPoints:
        raise ValueError(
            "miniTSNE size is %i samples per file which is low for current trainingSetSize which "
            "requires %i samples per file. "
            "Please decrease trainingSetSize or increase training_numPoints."
            % (parameters.training_numPoints, n_requested_per_dataset)
        )

    if parameters.waveletDecomp:
        trainingSetData = np.zeros((n_requested_per_dataset * L, numModes * numPeriods))
    else:
        trainingSetData = np.zeros((n_requested_per_dataset * L, numModes))
    trainingSetAmps = np.zeros((n_requested_per_dataset * L, 1))
    useIdx = np.ones((n_requested_per_dataset * L), dtype="bool")

    for i in tqdm(range(L)):
        print(
            "Finding training set contributions from data set %i/%i : \n%s"
            % (i + 1, L, projectionFiles[i])
        )

        currentIdx = np.arange(n_requested_per_dataset) + (i * n_requested_per_dataset)

        yData, signalData, _, signalAmps = file_embeddingSubSampling(
            projectionFiles[i], parameters
        )
        (
            trainingSetData[currentIdx, :],
            trainingSetAmps[currentIdx],
        ) = findTemplatesFromData(
            signalData, yData, signalAmps, n_requested_per_dataset, parameters, projectionFiles[i]
        )

        a = np.sum(trainingSetData[currentIdx, :], 1) == 0
        useIdx[currentIdx[a]] = False

    trainingSetData = trainingSetData[useIdx, :]
    trainingSetAmps = trainingSetAmps[useIdx]

    return trainingSetData, trainingSetAmps, projectionFiles


def subsampled_tsne_from_projections(parameters, results_directory):
    """
    Wrapper function for training set subsampling and mapping.
    """
    projection_directory = results_directory + "/Projections/"
    if parameters.method == "TSNE":
        if parameters.waveletDecomp:
            tsne_directory = results_directory + "/TSNE/"
        else:
            tsne_directory = results_directory + "/TSNE_Projections/"

        parameters.tsne_directory = tsne_directory

        parameters.tsne_readout = 50

        tSNE_method_old = parameters.tSNE_method
        if tSNE_method_old != "barnes_hut":
            print(
                "Setting tsne method to barnes_hut while subsampling for training set (for speedup)..."
            )
            parameters.tSNE_method = "barnes_hut"

    elif parameters.method == "UMAP":
        tsne_directory = results_directory + "/UMAP/"
        if not parameters.waveletDecomp:
            raise ValueError("Wavelet decomposition needed to run UMAP implementation.")
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")

    print("Finding Training Set")
    if not parameters.cache or not os.path.exists(tsne_directory + "training_data.mat"):
        trainingSetData, trainingSetAmps, _ = runEmbeddingSubSampling(
            projection_directory, parameters
        )
        if os.path.exists(tsne_directory):
            shutil.rmtree(tsne_directory)
            os.mkdir(tsne_directory)
        else:
            os.mkdir(tsne_directory)

        hdf5storage.write(
            data={"trainingSetData": trainingSetData},
            path="/",
            truncate_existing=True,
            filename=tsne_directory + "/training_data.mat",
            store_python_metadata=False,
            matlab_compatible=True,
        )

        hdf5storage.write(
            data={"trainingSetAmps": trainingSetAmps},
            path="/",
            truncate_existing=True,
            filename=tsne_directory + "/training_amps.mat",
            store_python_metadata=False,
            matlab_compatible=True,
        )

        del trainingSetAmps
    else:
        print(
            "Subsampled trainingSetData found, skipping minitSNE and running training tSNE"
        )
        with h5py.File(tsne_directory + "/training_data.mat", "r") as hfile:
            trainingSetData = hfile["trainingSetData"][:].T

    # %% Run t-SNE on training set
    if parameters.method == "TSNE":
        if tSNE_method_old != "barnes_hut":
            print("Setting tsne method back to to %s" % tSNE_method_old)
            parameters.tSNE_method = tSNE_method_old
        parameters.tsne_readout = 5
        print("Finding t-SNE Embedding for Training Set")

        trainingEmbedding = run_tSne(trainingSetData, parameters)
    elif parameters.method == "UMAP":
        print("Finding UMAP Embedding for Training Set")
        trainingEmbedding = run_UMAP(
            trainingSetData, parameters, metric=parameters.umapMetric
        )
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")
    hdf5storage.write(
        data={"trainingEmbedding": trainingEmbedding},
        path="/",
        truncate_existing=True,
        filename=tsne_directory + "/training_embedding.mat",
        store_python_metadata=False,
        matlab_compatible=True,
    )


"""Re-Embedding Code"""


def returnCorrectSigma_sparse(ds, perplexity, tol, maxNeighbors):
    highGuess = np.max(ds)
    lowGuess = 1e-12

    sigma = 0.5 * (highGuess + lowGuess)

    dsize = ds.shape
    sortIdx = np.argsort(ds)
    ds = ds[sortIdx[:maxNeighbors]]
    p = np.exp(-0.5 * np.square(ds) / sigma**2)
    p = p / np.sum(p)
    idx = p > 0
    H = np.sum(-np.multiply(p[idx], np.log(p[idx])) / np.log(2))
    P = 2**H

    if abs(P - perplexity) < tol:
        test = False
    else:
        test = True

    count = 0
    if ~np.isfinite(sigma):
        raise ValueError(
            "Starting sigma is %0.02f, highGuess is %0.02f "
            "and lowGuess is %0.02f" % (sigma, highGuess, lowGuess)
        )
    while test:
        if P > perplexity:
            highGuess = sigma
        else:
            lowGuess = sigma

        sigma = 0.5 * (highGuess + lowGuess)

        p = np.exp(-0.5 * np.square(ds) / sigma**2)
        if np.sum(p) > 0:
            p = p / np.sum(p)
        idx = p > 0
        H = np.sum(-np.multiply(p[idx], np.log(p[idx])) / np.log(2))
        P = 2**H

        if np.abs(P - perplexity) < tol:
            test = False

    out = np.zeros((dsize[0],))
    out[sortIdx[:maxNeighbors]] = p
    return sigma, out


def findListKLDivergences(data, data2):
    logData = np.log(data)

    entropies = -np.sum(np.multiply(data, logData), 1)
    del logData

    logData2 = np.log(data2)

    D = -np.dot(data, logData2.T)

    D = D - entropies[:, None]

    D = D / np.log(2)
    return D, entropies


def calculateKLCost(x, ydata, ps):
    d = np.sum(np.square(ydata - x), 1).T
    out = np.log(np.sum(1 / (1 + d))) + np.sum(np.multiply(ps, np.log(1 + d)))
    return out


def TDistProjs(
    i,
    q,
    perplexity,
    sigmaTolerance,
    maxNeighbors,
    trainingEmbedding,
    readout,
    waveletDecomp,
):
    if (i + 1) % readout == 0:
        t1 = time.time()
        print("\t\t Calculating Sigma Image #%5i" % (i + 1))
    _, p = returnCorrectSigma_sparse(q, perplexity, sigmaTolerance, maxNeighbors)

    if (i + 1) % readout == 0:
        print("\t\t Calculated Sigma Image #%5i" % (i + 1))

    idx2 = p > 0
    z = trainingEmbedding[idx2, :]
    maxIdx = np.argmax(p)
    a = np.sum(z * (p[idx2].T)[:, None], axis=0)

    guesses = [a, trainingEmbedding[maxIdx, :]]

    q = Delaunay(z)

    if (i + 1) % readout == 0:
        print("\t\t FminSearch Image #%5i" % (i + 1))

    b = np.zeros((2, 2))
    c = np.zeros((2,))
    flags = np.zeros((2,))

    if waveletDecomp:
        costfunc = calculateKLCost
    else:
        costfunc = calculateKLCost

    b[0, :], c[0], _, _, flags[0] = fmin(
        costfunc,
        x0=guesses[0],
        args=(z, p[idx2]),
        disp=False,
        full_output=True,
        maxiter=100,
    )
    b[1, :], c[1], _, _, flags[1] = fmin(
        costfunc,
        x0=guesses[1],
        args=(z, p[idx2]),
        disp=False,
        full_output=True,
        maxiter=100,
    )
    if (i + 1) % readout == 0:
        print(
            "\t\t FminSearch Done Image #%5i %0.02fseconds \n Flags are %s"
            % (i + 1, time.time() - t1, flags)
        )

    polyIn = q.find_simplex(b) >= 0

    if np.sum(polyIn) > 0:
        pp = np.where(polyIn)[0]
        mI = np.argmin(c[polyIn])
        mI = pp[mI]
        current_poly = True
    else:
        mI = np.argmin(c)
        current_poly = False
    if (i + 1) % readout == 0:
        print(
            "\t\t Simplex search done Image #%5i %0.02fseconds"
            % (i + 1, time.time() - t1)
        )
    exitFlags = flags[mI]
    current_guesses = guesses[mI]
    current = b[mI]
    tCosts = c[mI]
    current_meanMax = mI
    return current_guesses, current, tCosts, current_poly, current_meanMax, exitFlags


def findTDistributedProjections_fmin(data, trainingData, trainingEmbedding, parameters):
    readout = 100000
    sigmaTolerance = 1e-5
    perplexity = parameters.perplexity
    maxNeighbors = parameters.maxNeighbors
    batchSize = parameters.embedding_batchSize

    N = len(data)
    zValues = np.zeros((N, 2))
    zGuesses = np.zeros((N, 2))
    zCosts = np.zeros((N,))
    batches = np.ceil(N / batchSize).astype(int)
    inConvHull = np.zeros((N,), dtype=bool)
    meanMax = np.zeros((N,))
    exitFlags = np.zeros((N,))

    if parameters.numProcessors < 0:
        numProcessors = mp.cpu_count()
    else:
        numProcessors = parameters.numProcessors
    # ctx = mp.get_context('spawn')

    for j in range(batches):
        print("\t Processing batch #%4i out of %4i" % (j + 1, batches))
        idx = np.arange(batchSize) + j * batchSize
        idx = idx[idx < N]
        currentData = data[idx, :]

        if parameters.waveletDecomp:
            if np.sum(currentData == 0):
                print(
                    "Zeros found in wavelet data at following positions. Will replace then with 1e-12."
                )
                currentData[currentData == 0] = 1e-12

            print("\t Calculating distances for batch %4i" % (j + 1))
            t1 = time.time()
            D2, _ = findListKLDivergences(currentData, trainingData)
            print(
                "\t Calculated distances for batch %4i %0.02fseconds."
                % (j + 1, time.time() - t1)
            )
        else:
            print("\t Calculating distances for batch %4i" % (j + 1))
            t1 = time.time()
            D2 = distance.cdist(currentData, trainingData, metric="sqeuclidean")
            print(
                "\t Calculated distances for batch %4i %0.02fseconds."
                % (j + 1, time.time() - t1)
            )

        print("\t Calculating fminProjections for batch %4i" % (j + 1))
        t1 = time.time()
        pool = mp.Pool(numProcessors)
        outs = pool.starmap(
            TDistProjs,
            [
                (
                    i,
                    D2[i, :],
                    perplexity,
                    sigmaTolerance,
                    maxNeighbors,
                    trainingEmbedding,
                    readout,
                    parameters.waveletDecomp,
                )
                for i in range(len(idx))
            ],
        )

        zGuesses[idx, :] = np.concatenate([out[0][:, None] for out in outs], axis=1).T
        zValues[idx, :] = np.concatenate([out[1][:, None] for out in outs], axis=1).T
        zCosts[idx] = np.array([out[2] for out in outs])
        inConvHull[idx] = np.array([out[3] for out in outs])
        meanMax[idx] = np.array([out[4] for out in outs])
        exitFlags[idx] = np.array([out[5] for out in outs])
        pool.close()
        pool.join()
        print(
            "\t Processed batch #%4i out of %4i in %0.02fseconds.\n"
            % (j + 1, batches, time.time() - t1)
        )

    zValues[~inConvHull, :] = zGuesses[~inConvHull, :]

    return zValues, zCosts, zGuesses, inConvHull, meanMax, exitFlags


import multiprocessing


def findEmbeddings(
    projections, trainingData, trainingEmbedding, parameters, projectionFile
):
    """
    findEmbeddings finds the optimal embedding of a data set into a previously
    found t-SNE embedding.
    :param projections:  N x (pcaModes x numPeriods) array of projection values.
    :param trainingData: Nt x (pcaModes x numPeriods) array of wavelet amplitudes containing Nt data points.
    :param trainingEmbedding: Nt x 2 array of embeddings.
    :param parameters: motionmapperpy Parameters dictionary.
    :return: zValues : N x 2 array of embedding results, outputStatistics : dictionary containing other parametric
    outputs.
    """
    d = projections.shape[1]
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods

    if parameters.waveletDecomp:
        print("Finding Wavelets")
        if not os.path.exists(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        ):
            data, f = mm_findWavelets(projections, numModes, parameters)
        else:
            print("\n Loading wavelets")
            with h5py.File(
                f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
                "r",
            ) as f:
                data = f["wavelets"][:]
                data[~np.isfinite(data)] = 1e-12
                data[data == 0] = 1e-12
        print("\n Loaded wavelets")
    else:
        print("Using projections for tSNE. No wavelet decomposition.")
        f = 0
        data = projections

    raw_wavelets = data.copy()
    # Apply condition on wavelets
    data_sum = np.sum(raw_wavelets, 1)
    idx_valid = data_sum > MIN_POWER  # indices of valid points

    # Only valid points are embedded
    valid_data = data[idx_valid]
    print(f"Valid data shape: {valid_data.shape} out of {data.shape}")

    data = data / np.sum(data, 1)[:, None]

    print("Finding Embeddings")
    t1 = time.time()
    if parameters.method == "TSNE":
        (
            zValues_temp,
            zCosts,
            zGuesses,
            inConvHull,
            meanMax,
            exitFlags,
        ) = findTDistributedProjections_fmin(
            valid_data, trainingData, trainingEmbedding, parameters
        )

        outputStatistics_temp = edict()
        outputStatistics_temp.zCosts = zCosts
        outputStatistics_temp.f = f
        outputStatistics_temp.numModes = numModes
        outputStatistics_temp.zGuesses = zGuesses
        outputStatistics_temp.inConvHull = inConvHull
        outputStatistics_temp.meanMax = meanMax
        outputStatistics_temp.exitFlags = exitFlags
    elif parameters.method == "UMAP":
        # Split valid data into chunks for parallel processing
        n_jobs=parameters.numProcessors
        n_jobs=1
        valid_data_chunks = np.array_split(valid_data, n_jobs)
        print(f"Using {n_jobs} processors for embedding")

        if n_jobs > 1:
            pool = multiprocessing.Pool(processes=n_jobs)
            # Parallelize the UMAP transform for valid data chunks
            results = pool.starmap(
                umap_transform, [(chunk, parameters) for chunk in valid_data_chunks]
            )
        else:
            results=[]
            for chunk in valid_data_chunks:
                results.append(
                    umap_transform(chunk, parameters)
                )

        # Combine the results
        zValues_temp_list, trainparams_list = zip(*results)
        zValues_temp = np.concatenate(zValues_temp_list)
        trainparams = trainparams_list[
            0
        ]  # Assuming trainparams are the same for all chunks

        outputStatistics = edict()
        outputStatistics.training_mean = trainparams[0]
        outputStatistics.training_scale = trainparams[1]
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")

        # Initialize zValues with 'NA' for all points
    zValues = np.full((data.shape[0], 2), np.nan)

    # Assign computed embeddings to valid points
    zValues[idx_valid] = zValues_temp

    del data
    print("Embeddings found in %0.02f seconds." % (time.time() - t1))

    return zValues, outputStatistics


def umap_transform(data, parameters):
    umapfolder = parameters["projectPath"] + "/UMAP/"
    if parameters.useGPU>=0:
        umap_file=umapfolder + "umap.joblib"
        um = joblib.load(umap_file)
        print("Using GPU UMAP")

    else:
        umap_file=umapfolder + "umap.model"
        with open(umap_file, "rb") as f:
            um = pickle.load(f)    
        print("Using CPU UMAP")

    trainparams = np.load(umapfolder + "_trainMeanScale.npy", allow_pickle=True)
    embed_negative_sample_rate = parameters["embed_negative_sample_rate"]
    um.negative_sample_rate = embed_negative_sample_rate
    zValues = um.transform(data)
    zValues = zValues - trainparams[0]
    zValues = zValues * trainparams[1]
    return zValues, trainparams


def file_embeddingSubSampling_batch(projectionFile, parameters):
    numPoints = parameters.training_numPoints

    with h5py.File(projectionFile, "r") as hfile:
        projections_shape = hfile["projections"][:].T.shape

    # projections_shape = timepoints x n_pcs
    # TODO: Make this more general
    edge_file = (
        "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/sample_tracks/edge/"
        + pathlib.Path(projectionFile).stem.split("-")[0]
        + "_edge.mat"
    )
    if os.path.exists(edge_file):
        print(f"Using edge file: {edge_file}")

        # fly_num = int(pathlib.Path(projectionFile).stem.split("-")[3].split("_")[0])
        with h5py.File(edge_file, "r") as hfile:
            edge_mask = np.append([False], hfile["edger"][:].T[:, 0].astype(bool))
            # edge_mask = np.append([False], hfile["edger"][:].T[:, fly_num].astype(bool))
    else:
        logging.warning(f"Edge file {edge_file} not found")
        edge_file=None
        edge_mask = np.zeros((projections_shape[0]))

    print(f"projection file: {projectionFile}")
    print(f"edge file: {edge_file}")

    print(f"projections shape: {projections_shape}")
    print(f"edge shape: {edge_mask.shape}")
    # edge_mask = edge_mask[: projections_shape[0]]
    print(f"Frac on edge: {np.sum(edge_mask)/projections_shape[0]}")

    missingness_file = f"{parameters.projectPath}/Ego/{Path(projectionFile).stem}.h5"
    with h5py.File(
        missingness_file,
        "r",
    ) as hfile:
        print(f"projection file: {projectionFile}")
        print(f"missingness file: {missingness_file}")
        missingness_mask = hfile["missing_data_indices"][:].T.astype(bool)

        print(f"projections shape: {missingness_mask.shape}")
        print(f"missingness shape: {edge_mask.shape}")
        print(f"Frac missing: {np.sum(missingness_mask)/projections_shape[0]}")
    
    if projections_shape[0] < numPoints:
        raise ValueError(
            "Training number of points for miniTSNE is greater than # samples in some files. Please "
            "adjust it to %i or lower" % (projections_shape[0])
        )

    N = projections_shape[0]
    skipLength = np.floor(N / numPoints).astype(int)
    if skipLength == 0:
        skipLength = 1
        numPoints = N

    print(f"Subsampling {N} points to {numPoints} points")

    if parameters.waveletDecomp:
        # TODO: Don't be stupid. Load the wavelets once and then subsample them.
        with h5py.File(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
            "r",
        ) as f:
            wlets = f["wavelets"][:]
            # timepoints x (frequencies * n_pcs)
            print(f"wavelets shape: {wlets.shape}")
            sum_mask = np.sum(wlets, axis=1) < MIN_POWER
            print(f"sum_mask shape: {sum_mask.shape}")
            print(
                f"Fraction with amp lower than {MIN_POWER}: {np.sum(sum_mask)/projections_shape[0]}"
            )
        signalIdx = np.indices((projections_shape[0],))[0]
        masks=[
            # true for any timepoint where the fly was in the edge
            edge_mask,
            # true for any timepoint where the fly was missing
            missingness_mask,
            # true for any timepoint where the combined power of all wavelets is under MIN_POWER
            sum_mask
        ]
        # print(f"edge_mask shape: {edge_mask.shape}")
        signalIdx = apply_mask(signalIdx, masks, np.any)
        signalIdx = subsample(signalIdx, numPoints, skipLength)


        print(f"Final signalIdx: {signalIdx[0:10]}")
        print(f"Final signalIdx.shape: {signalIdx.shape}")
        # print("\t Calculating Wavelets")
        # print(
        #     f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        # )
        print("\n Loading wavelets")
        with h5py.File(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
            "r",
        ) as f:
            data = f["wavelets"][sorted(signalIdx)]
        print("\n Loaded wavelets")

        # get templates and real training data

        with open("list_of_working_files.txt", "a") as myfile:
            myfile.write(f"{projectionFile}\n")
        with open("list_of_working_files_length.txt", "a") as myfile:
            myfile.write(f"{signalIdx.shape[0]}\n")

        print("\n Subsampled wavelets")
        if not os.path.exists(
            f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat"
        ):
            with h5py.File(
                f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat",
                "w",
            ) as f:
                f.create_dataset("signaldata", data=data, compression="lzf")
        else:
            print("File already exists")
    return


def subsample(index, number_points, skip_length):

    # Subset to remove edge calls        
    if index.shape[0] < number_points:
        print("Warning: Not enough points to sample. Using all points")
        if skip_length == 0:
            skip_length = 1
            number_points = index.shape[0]
        print(f"Final signalIdx.shape: {index.shape}")
    else:
        print(f"Found {index.shape[0]} points to sample")
        skip_length = np.floor(index.shape[0] / number_points).astype(int)
        index = index[0 : int(0 + (number_points) * skip_length) : skip_length]
    return index

def apply_mask(index, masks, f):
    """
    Subset an index using a list of masks combined using f
    """
    print(f"index shape: {index.shape}")
    mask = f(np.vstack(masks).T, axis=1)

    print(f"mask shape: {mask.shape}")
    index = index[[not mask_ele for mask_ele in mask]]
    return index

def runEmbeddingSubSampling_batch(projectionDirectory, parameters, i):
    """
    runEmbeddingSubSampling generates a training set given a set of .mat files.

    :param projectionDirectory: directory path containing .mat projection files.
    Each of these files should contain an N x pcaModes variable, 'projections'.
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
        trainingSetData -> normalized wavelet training set
                           (N x (pcaModes*numPeriods) )
        trainingSetAmps -> Nx1 array of training set wavelet amplitudes
        projectionFiles -> list of files in 'projectionDirectory'
    """
    parameters = setRunParameters(parameters)
    projectionFiles = glob.glob(projectionDirectory + "/*pcaModes.mat")
    projectionFiles = natsort.natsorted(projectionFiles)

    file_embeddingSubSampling_batch(projectionFiles[i], parameters)


def subsampled_tsne_from_projections_batch(parameters, results_directory, i):
    """
    Wrapper function for training set subsampling and mapping.
    """
    projection_directory = results_directory + "/Projections/"
    if parameters.method == "TSNE":
        if parameters.waveletDecomp:
            tsne_directory = results_directory + "/TSNE/"
        else:
            tsne_directory = results_directory + "/TSNE_Projections/"

        parameters.tsne_directory = tsne_directory

        parameters.tsne_readout = 50

        tSNE_method_old = parameters.tSNE_method
        if tSNE_method_old != "barnes_hut":
            print(
                "Setting tsne method to barnes_hut while subsampling for training set (for speedup)..."
            )
            parameters.tSNE_method = "barnes_hut"

    elif parameters.method == "UMAP":
        tsne_directory = results_directory + "/UMAP/"
        if not parameters.waveletDecomp:
            raise ValueError("Wavelet decomposition needed to run UMAP implementation.")
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")

    print("Finding Training Set")
    # if not os.path.exists(tsne_directory + "training_data.mat"):
    runEmbeddingSubSampling_batch(projection_directory, parameters, i)
