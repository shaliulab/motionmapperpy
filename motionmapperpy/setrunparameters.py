from easydict import EasyDict as edict


def setRunParameters(parameters=None):
    """
    Get parameter dictionary for running motionmapperpy.
    :param parameters: Existing parameter dictionary, defaults will be filled for missing keys.
    :return: Parameter dictionary.
    """
    if isinstance(parameters, dict):
        parameters = edict(parameters)
    else:
        parameters = edict()

    """# %%%%%%%% General Parameters %%%%%%%%"""

    projectPath = "mmpy_lts_1d"

    # %number of processors to use in parallel code
    numProcessors = 16

    useGPU = -1  # -1 for CPU, 0 for first GPU, 1 for second GPU, etc.
# naps-track --slp-path 1h_example.slp --h5-path 1min_example.analysis.h5 --video-path 20220823-cut-to-1200to1300_libx264.mp4 --start-frame 0 --end-frame 1199 --aruco-marker-set DICT_5X5_50  --output-path example_output.analysis.h5 --aruco-error-correction-rate 1  --aruco-adaptive-thresh-constant 3 --aruco-adaptive-thresh-win-size-max 30 --aruco-adaptive-thresh-win-size-step 3 --aruco-perspective-rm-ignored-margin 0.13 --aruco-adaptive-thresh-win-size-min 3 --half-rolling-window-size 50 --tag-node 0
    method = "TSNE"  # or 'UMAP'

    """%%%%%%%% Wavelet Parameters %%%%%%%%"""
    # %Whether to do wavelet decomposition, if False then use normalized projections for tSNE embedding.
    waveletDecomp = True

    # %number of wavelet frequencies to use
    numPeriods = 25

    # dimensionless Morlet wavelet parameter
    omega0 = 5

    # sampling frequency (Hz)
    samplingFreq = 100

    # minimum frequency for wavelet transform (Hz)
    minF = 1

    # maximum frequency for wavelet transform (Hz)
    maxF = 50

    """%%%%%%%% t-SNE Parameters %%%%%%%%"""
    # Global tSNE method - 'barnes_hut' or 'exact'
    tSNE_method = "barnes_hut"

    # %2^H (H is the transition entropy)
    perplexity = 32

    # %embedding batchsize
    embedding_batchSize = 128000

    # %maximum number of iterations for the Nelder-Mead algorithm
    maxOptimIter = 500

    # %number of points in the training set
    trainingSetSize = 64000

    # %number of neigbors to use when re-embedding
    maxNeighbors = 200

    # %local neighborhood definition in training set creation
    kdNeighbors = 5

    # %t-SNE training set perplexity
    training_perplexity = 20

    # %number of points to evaluate in each training set file
    training_numPoints = 32000

    # %minimum training set template length
    minTemplateLength = 1

    """%%%%%%%% UMAP Parameters %%%%%%%%"""
    # Size of local neighborhood for UMAP.
    n_neighbors = 15

    # Negative sample rate while training.
    train_negative_sample_rate = 5

    # Negative sample rate while embedding new data.
    embed_negative_sample_rate = 1

    # Minimum distance between neighbors.
    min_dist = 0.1

    # UMAP output dimensions.
    umap_output_dims = 2

    # Number of training epochs.
    n_training_epochs = 100

    # Embedding rescaling parameter.
    rescale_max = 100

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    if not "numProcessors" in parameters.keys():
        parameters.numProcessors = numProcessors

    if not "numPeriods" in parameters.keys():
        parameters.numPeriods = numPeriods

    if not "omega0" in parameters.keys():
        parameters.omega0 = omega0

    if not "samplingFreq" in parameters.keys():
        parameters.samplingFreq = samplingFreq

    if not "minF" in parameters.keys():
        parameters.minF = minF

    if not "maxF" in parameters.keys():
        parameters.maxF = maxF

    if not "tSNE_method" in parameters.keys():
        parameters.tSNE_method = tSNE_method

    if not "perplexity" in parameters.keys():
        parameters.perplexity = perplexity

    if not "embedding_batchSize" in parameters.keys():
        parameters.embedding_batchSize = embedding_batchSize

    if not "maxOptimIter" in parameters.keys():
        parameters.maxOptimIter = maxOptimIter

    if not "trainingSetSize" in parameters.keys():
        parameters.trainingSetSize = trainingSetSize

    if not "maxNeighbors" in parameters.keys():
        parameters.maxNeighbors = maxNeighbors

    if not "kdNeighbors" in parameters.keys():
        parameters.kdNeighbors = kdNeighbors

    if not "training_perplexity" in parameters.keys():
        parameters.training_perplexity = training_perplexity

    if not "training_numPoints" in parameters.keys():
        parameters.training_numPoints = training_numPoints

    if not "minTemplateLength" in parameters.keys():
        parameters.minTemplateLength = minTemplateLength

    if not "waveletDecomp" in parameters.keys():
        parameters.waveletDecomp = waveletDecomp

    if not "useGPU" in parameters.keys():
        parameters.useGPU = useGPU

    if not "n_neighbors" in parameters.keys():
        parameters.n_neighbors = n_neighbors

    if not "train_negative_sample_rate" in parameters.keys():
        parameters.train_negative_sample_rate = train_negative_sample_rate

    if not "embed_negative_sample_rate" in parameters.keys():
        parameters.embed_negative_sample_rate = embed_negative_sample_rate

    if not "min_dist" in parameters.keys():
        parameters.min_dist = min_dist

    if not "umap_output_dims" in parameters.keys():
        parameters.umap_output_dims = umap_output_dims

    if not "n_training_epochs" in parameters.keys():
        parameters.n_training_epochs = n_training_epochs

    if not "rescale_max" in parameters.keys():
        parameters.rescale_max = rescale_max

    if not "method" in parameters.keys():
        parameters.method = method

    if not "projectPath" in parameters.keys():
        parameters.projectPath = projectPath

    return parameters
