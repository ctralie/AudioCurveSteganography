import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import time
from numba import jit

def dpoint2pointcloud(X, i, metric):
    """
    Return the distance from the ith point in a Euclidean point cloud
    to the rest of the points
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of data 
    i: int
        The index of the point from which to return all distances
    metric: string or callable
        The metric to use when calculating distance between instances in a 
        feature array
    """
    ds = pairwise_distances(X, X[i, :][None, :], metric=metric).flatten()
    ds[i] = 0
    return ds


def get_greedy_perm(X, n_perm=None, distance_matrix=False, metric="euclidean"):
    """
    Compute a furthest point sampling permutation of a set of points
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix
    distance_matrix: bool
        Indicator that X is a distance matrix, if not we compute 
        distances in X using the chosen metric.
    n_perm: int
        Number of points to take in the permutation
    metric: string or callable
        The metric to use when calculating distance between instances in a 
        feature array
    Returns
    -------
    idx_perm: ndarray(n_perm)
        Indices of points in the greedy permutation
    lambdas: ndarray(n_perm)
        Covering radii at different points
    dperm2all: ndarray(n_perm, n_samples)
        Distances from points in the greedy permutation to points
        in the original point set
    """
    if not n_perm:
        n_perm = X.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    lambdas = np.zeros(n_perm)
    if distance_matrix:
        dpoint2all = lambda i: X[i, :]
    else:
        dpoint2all = lambda i: dpoint2pointcloud(X, i, metric)
    ds = dpoint2all(0)
    dperm2all = [ds]
    for i in range(1, n_perm):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        lambdas[i - 1] = ds[idx]
        dperm2all.append(dpoint2all(idx))
        ds = np.minimum(ds, dperm2all[-1])
    lambdas[-1] = np.max(ds)
    dperm2all = np.array(dperm2all)
    return (idx_perm, lambdas, dperm2all)

def bosch_grid_stipple(I, res, gamma, contrast_boost):
    """
    An implementation of the method of [1]

    [1] Robert Bosch and Adrianne Herman. Continuous line drawings via the
        traveling salesman problem. Operations research letters, 32(4):302–303, 2004.
    
    Parameters
    ----------
    I: ndarray(M, N)
        A grayscale image
    
    res: int
        Resolution of the grid squares in which to stipple
    gamma: float
        Number of cities per square
    contrast_boost: bool
        Whether to do a contrast boost
    """
    if np.max(I) > 1:
        I = I/255.0
    M = int(I.shape[0]/res)
    N = int(I.shape[1]/res)
    X = np.array([])
    for i in range(M):
        for j in range(N):
            mu = np.mean(I[i*res:(i+1)*res, j*res:(j+1)*res])
            g = int(gamma-np.floor((gamma+1)*mu))
            if contrast_boost:
                g = int(g**2/3)
            x = res*np.random.rand(g, 2) + np.array([[j*res, i*res]])
            if X.size == 0:
                X = x
            else:
                X = np.concatenate((X, x), axis=0)
    return X


def get_weights(I, thresh, p=1, canny_sigma=0, edge_weight=1):
    """
    Create pre-pixel weights based on image brightness

    Parameters
    ----------
    I: ndarray(M, N, ..)
        An rgb/ grayscale image
    thresh: float
        Amount above which to make a point 1
    p: float
        Contrast boost, apply weights^(1/p)
    canny_sigma: float
        If >0, use a canny edge detector with this standard deviation
    edge_weight: float
        Weight to apply to edges, relative to the background
    
    Returns
    -------
    ndarray(M, N)
        The weights of each pixel, in the range [0, 1]
    """
    if np.max(I) > 1:
        I = I/255
    if len(I.shape) > 2:
        I = 0.2125*I[:, :, 0] + 0.7154*I[:, :, 1] + 0.0721*I[:, :, 2]
    ## Step 1: Get weights and initialize random point distribution
    weights = np.array(I)
    if np.max(weights) > 1:
        weights /= 255
    weights = np.minimum(weights, thresh)
    weights -= np.min(weights)
    if np.max(weights) == 0:
        weights = 0*weights
    else:
        weights /= np.max(weights)
        weights = 1-weights
        weights = weights**(1/p)
    if canny_sigma > 0:
        from skimage import feature
        if edge_weight > 1:
            weights /= edge_weight
        edges = feature.canny(I, sigma=canny_sigma)
        weights[edges > 0] = 1
    return weights


def rejection_sample_by_density(weights, target_points):
    """
    Sample points according to a particular density, by rejection sampling

    Parameters
    ----------
    ndarray(M, N)
        The weights of each pixel, in the range [0, 1]
    target_points: int
        The number of desired samples
    
    Returns
    -------
    uv: ndarray(N, 2)
        Location of point samples, in units of pixels
    """
    uv = np.zeros((target_points, 2))
    idx = 0
    while idx < target_points:
        print(idx)
        I = np.random.rand(10*target_points)*(weights.shape[0]-1)
        J = np.random.rand(10*target_points)*(weights.shape[1]-1)
        P = np.random.rand(10*target_points)
        for i, j, p in zip(I, J, P):
            weight = weights[int(np.floor(i)), int(np.floor(j))]
            if p < weight:
                uv[idx, :] = [i, j]
                idx += 1
                if idx == target_points:
                    return uv
    return uv

def stochastic_universal_sample(weights, target_points, jitter=0.1):
    """
    Sample pixels according to a particular density using 
    stochastic universal sampling

    Parameters
    ----------
    ndarray(M, N)
        The weights of each pixel, in the range [0, 1]
    target_points: int
        The number of desired samples
    jitter: float
        Perform a jitter with this standard deviation of a pixel
    
    Returns
    -------
    ndarray(N, 2)
        Location of point samples
    """
    choices = np.zeros(target_points, dtype=np.int64)
    w = np.zeros(weights.size+1)
    order = np.random.permutation(weights.size)
    w[1::] = weights.flatten()[order]
    w = w/np.sum(w)
    w = np.cumsum(w)
    p = np.random.rand() # Cumulative probability index, start off random
    idx = 0
    for i in range(target_points):
        while idx < weights.size and not (p >= w[idx] and p < w[idx+1]):
            idx += 1
        idx = idx % weights.size
        choices[i] = order[idx]
        p = (p + 1/target_points) % 1
    X = np.array(list(np.unravel_index(choices, weights.shape)), dtype=float).T
    if jitter > 0:
        X += jitter*np.random.randn(X.shape[0], 2)
    return X

def get_centroids_edt(X, weights):
    """
    Compute weighted centroids of Voronoi regions of points in X

    Parameters
    ----------
    X: ndarray(n_points, 2)
        Points locations
    weights: ndarray(M, N)
        Weights to use at each pixel in the Voronoi image
    
    Returns
    -------
    ndarray(<=n_points, 2)
        Points moved to their centroids.  Note that some points may die
        off if no pixel is nearest to them
    """
    from scipy.ndimage import distance_transform_edt
    from scipy import sparse
    ## Step 1: Comput Euclidean Distance Transform
    mask = np.ones_like(weights)
    X = np.array(np.round(X), dtype=np.int64)
    mask[X[:, 0], X[:, 1]] = 0
    _, inds = distance_transform_edt(mask, return_indices=True)
    ## Step 2: Take weighted average of all points that have the same
    ## label in the euclidean distance transform, using scipy's sparse
    ## to quickly add up weighted coordinates of all points with the same label
    inds = inds[0, :, :]*inds.shape[2] + inds[1, :, :]
    inds = inds.flatten()
    N = len(np.unique(inds))
    idx2idx = -1*np.ones(inds.size)
    idx2idx[np.unique(inds)] = np.arange(N)
    inds = idx2idx[inds]
    ii, jj = np.meshgrid(np.arange(weights.shape[0]), np.arange(weights.shape[1]), indexing='ij')
    cols_i = (weights*ii).flatten()
    cols_j = (weights*jj).flatten()
    num_i = sparse.coo_matrix((cols_i, (inds, np.zeros(inds.size, dtype=np.int64))), shape=(N, 1)).toarray().flatten()
    num_j = sparse.coo_matrix((cols_j, (inds, np.zeros(inds.size, dtype=np.int64))), shape=(N, 1)).toarray().flatten()
    denom = sparse.coo_matrix((weights.flatten(), (inds, np.zeros(inds.size, dtype=np.int64))), shape=(N, 1)).toarray().flatten()
    num_i = num_i[denom > 0]
    num_j = num_j[denom > 0]
    denom = denom[denom > 0]
    return np.array([num_i/denom, num_j/denom]).T

def voronoi_stipple(I, thresh, target_points, p=1, canny_sigma=0, edge_weight=1, n_iters=10, do_plot=False):
    """
    An implementation of the method of [2]

    [2] Adrian Secord. Weighted Voronoi Stippling
    
    Parameters
    ----------
    I: ndarray(M, N, ..)
        An rgb/ grayscale image
    thresh: float
        Amount above which to make a point 1
    p: float
        Contrast boost, apply weights^(1/p)
    canny_sigma: float
        If >0, use a canny edge detector with this standard deviation
    edge_weight: float
        Weight to apply to edges, relative to the background
    n_iters: int
        Number of iterations
    do_plot: bool
        Whether to plot each iteration
    
    Returns
    -------
    ndarray(<=target_points, 2)
        An array of the stipple pattern, with x coordinates along the first
        column and y coordinates along the second column.
        Note that the number of points may be slightly less than the requested
        points due to density filtering or resolution limits of the Voronoi computation
    """
    if np.max(I) > 1:
        I = I/255
    if len(I.shape) > 2:
        I = 0.2125*I[:, :, 0] + 0.7154*I[:, :, 1] + 0.0721*I[:, :, 2]
    ## Step 1: Get weights and initialize random point distribution
    ## via rejection sampling
    weights = get_weights(I, thresh, p, canny_sigma=canny_sigma, edge_weight=edge_weight)
    X = stochastic_universal_sample(weights, target_points)
    X = np.array(np.round(X), dtype=int)
    X[X[:, 0] >= weights.shape[0], 0] = weights.shape[0]-1
    X[X[:, 1] >= weights.shape[1], 1] = weights.shape[1]-1

    ## Step 2: Repeatedly re-compute centroids of Voronoi regions
    if do_plot:
        plt.figure(figsize=(10, 10))
    for it in range(n_iters):
        if do_plot:
            plt.clf()
            plt.scatter(X[:, 1], X[:, 0], 1)
            plt.gca().invert_yaxis()
            plt.xlim([0, weights.shape[1]])
            plt.ylim([weights.shape[0], 0])
            plt.savefig("Voronoi{}.png".format(it), facecolor='white')
        X = get_centroids_edt(X, weights)

    X[:, 0] = I.shape[0]-X[:, 0]
    return np.fliplr(X)

def get_char_freetype(face, c):
    face.load_char(c)
    X = []
    def move_to(p, _):
        X.append([p.x, p.y])
    def segment_to(*args):
        *args, _ = args
        X.extend([[p.x, p.y] for p in args])
    face.glyph.outline.decompose(None, move_to, segment_to, segment_to, segment_to)
    return np.array(X, dtype=float)

def get_string_freetype(face, s, samples_per_char, space=1):
    """
    Compute the points on a TTF font curve
    Requires installation of freetype-py package

    Parameters
    ----------
    face: freetype.Face
        Face object
    s: string
        String to create
    samples_per_char: int
        Number of samples per letter in the string
    space: float
        Spacing between characters
    """
    from curvature import arclen_resample_linear
    # Calibrate width with @ symbol
    Xw = get_char_freetype(face, ".")
    width = np.max(Xw[:, 0]) - np.min(Xw[:, 0])
    X = np.zeros((0, 2))
    offset = 0
    for c in s:
        Xc = get_char_freetype(face, c)
        cwidth = width
        if Xc.size > 0:
            Xc[:, 0] += offset
            Xc = arclen_resample_linear(Xc, samples_per_char)
            cwidth = np.max(Xc[:, 0])-np.min(Xc[:, 0])
            X = np.concatenate((X, Xc), axis=0)
        offset += cwidth + space*width
    return X