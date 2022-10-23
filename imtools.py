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
    ## Step 1: Get weights and initialize random point distributin
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
    ndarray(N, 2)
        Location of point samples
    """
    X = np.zeros((target_points, 2))
    idx = 0
    while idx < target_points:
        print(idx)
        I = np.random.rand(10*target_points)*(weights.shape[0]-1)
        J = np.random.rand(10*target_points)*(weights.shape[1]-1)
        P = np.random.rand(10*target_points)
        for i, j, p in zip(I, J, P):
            weight = weights[int(np.floor(i)), int(np.floor(j))]
            if p < weight:
                X[idx, :] = [i, j]
                idx += 1
                if idx == target_points:
                    return X
    return X

@jit(nopython=True)
def get_centroids(mask, N, weights):
    """
    Return the weighted centroids in a mask
    """
    nums = np.zeros((N, 2))
    denoms = np.zeros(N)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            idx = int(mask[i, j])
            weight = weights[i, j]
            nums[idx, 0] += weight*i
            nums[idx, 1] += weight*j
            denoms[idx] += weight
    nums = nums[denoms > 0, :]
    denoms = denoms[denoms > 0]
    return nums, denoms

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
    """
    from scipy.ndimage import distance_transform_edt
    import time
    if np.max(I) > 1:
        I = I/255
    if len(I.shape) > 2:
        I = 0.2125*I[:, :, 0] + 0.7154*I[:, :, 1] + 0.0721*I[:, :, 2]
    ## Step 1: Get weights and initialize random point distribution
    ## via rejection sampling
    weights = get_weights(I, thresh, p, canny_sigma, edge_weight)
    X = rejection_sample_by_density(weights, target_points)
    X = np.array(np.round(X), dtype=int)
    X[X[:, 0] >= weights.shape[0], 0] = weights.shape[0]-1
    X[X[:, 1] >= weights.shape[1], 1] = weights.shape[1]-1

    if do_plot:
        plt.figure(figsize=(12, 6))
    for it in range(n_iters):
        if do_plot:
            plt.clf()
            plt.subplot(121)
            plt.imshow(weights)
            plt.subplot(122)
            plt.scatter(X[:, 1], X[:, 0], 4)
            plt.gca().invert_yaxis()
            plt.xlim([0, weights.shape[1]])
            plt.ylim([weights.shape[0], 0])
            plt.savefig("Voronoi{}.png".format(it), facecolor='white')
        
        mask = np.ones_like(weights)
        X = np.array(np.round(X), dtype=int)
        mask[X[:, 0], X[:, 1]] = 0

        _, inds = distance_transform_edt(mask, return_indices=True)
        ind2num = {}
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                coord = (inds[0, i, j], inds[1, i, j])
                if not coord in ind2num:
                    ind2num[coord] = len(ind2num)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                coord = (inds[0, i, j], inds[1, i, j])
                mask[i, j] = ind2num[coord]
        nums, denoms = get_centroids(mask, len(ind2num), weights)
        X = nums/denoms[:, None]
    X[:, 0] = I.shape[0]-X[:, 0]
    return np.fliplr(X)
