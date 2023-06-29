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
    choices = np.zeros(target_points, dtype=int)
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
    if np.sum(weights) == 0:
        print("WARNING: No significant features found")
        return np.array([[]])
    X = stochastic_universal_sample(weights, target_points)
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


def splat_voronoi_image(xy, X, xpix, ypix, I, ICPU, n_neighbs, temperature=20):
    """
    Compute the Voronoi image associated to a set of xyrgb points in [0, 1]^5

    Parameters
    ----------
    x: torch.tensor(n_points, 2)
        Coordinates of voronoi sites
    X: ndarray(M*N, 2)
        Pixel locations
    xpix: torch.tensor(M*N)
        Pixel x coordinates
    ypix: torch.tensor(M*N)
        Pixel y coordinates
    I: torch.tensor(M*N, 3)
        Flattened image, in torch
    ICPU: np.array(M*N, 3)
        Flattened image, in numpy
    n_neighbs: int
        Number of sites to use as nearest neighbors for each point
    temperature: float
        Temperature to use in softmax for neighbor weights
    
    Returns
    -------
    J: torch.tensor(M*N, 3)
        Voronoi image
    rgb: torch.tensor(n_points, 3)
        Voronoi site colors
    """
    import torch
    from scipy import sparse
    from scipy.sparse import linalg as slinalg
    from scipy.spatial import KDTree
    ## Step 1: Compute distance from all pixels to landmarks
    tree = KDTree(xy.detach().cpu().numpy())
    _, idx = tree.query(X, k=n_neighbs) # Nearest landmark indices to pixels
    
    ## Step 2: Based on the landmark locations, compute the weight influence
    ## of each landmark on each pixel using softmax
    weights = torch.zeros(I.shape[0], n_neighbs).to(I)
    for i in range(n_neighbs):
        dx = xy[idx[:, i], 0] - xpix
        dy = xy[idx[:, i], 1] - ypix
        weights[:, i] = 1/(dx*dx + dy*dy)
    mx = torch.max(weights, dim=1).values
    mx = mx.view(mx.numel(), 1)
    weights = torch.exp(temperature*weights/mx)
    weights = weights/torch.sum(weights, dim=1, keepdims=True)
    
    ## Step 3: Compute the color of each landmark according to the weights
    ## by solving a sparse least squares system
    pixidx = (np.arange(idx.shape[0])[:, None]*np.ones((1, n_neighbs))).flatten()
    orig_shape = idx.shape
    idx = idx.flatten()
    weights = weights.flatten()
    A = sparse.coo_matrix((weights.detach().cpu().numpy(), (pixidx, idx)), shape=(I.shape[0], xy.shape[0]))
    rgb = np.ones((xy.shape[0], 3))
    for channel in range(3):
        rgb[:, channel] = slinalg.lsqr(A, ICPU[:, channel])[0]
    rgb = torch.from_numpy(rgb).to(xy)
    
    ## Step 4: Splat the colors from each landmark over the pixels within their influence
    idx = np.reshape(idx, orig_shape)
    weights = torch.reshape(weights, orig_shape)
    J = torch.zeros(I.shape).to(I)
    for i in range(n_neighbs):
        J += rgb[idx[:, i], :]*(weights[:, i].view(weights.shape[0], 1))
    return J, rgb


def get_voronoi_image(I, device, n_points, n_neighbs=2, n_iters=50, lr=2e-2, do_weight_plot=False, plot_iter_interval=0, verbose=False):
    """
    Compute a set of Voronoi sites best fit to an image

    Parameters
    ----------
    I: ndarray(M, N, [optional channel])
        Original image
    n_points: int
        Number of Voronoi sites
    n_neighbs: int
        Number of nearest neighbors to use in the voronoi diagram (default 2)
    n_iters: int
        Number of iterations of gradient descent
    lr: float
        Learning rate
    do_weight_plot: bool
        Whether to make a plot showing the weights
    plot_iter_interval: int
        If > 0, make a plot showing the gradient descent every time this number of iterations passes
    verbose: bool
        Whether to plot information about the iterations

    Returns
    -------
    J: torch.tensor(M*N, 3)
        Voronoi image
    xy: torch.tensor(n_points, 2)
        x/y coordinates of Voronoi sites
    rgb: torch.tensor(n_points, 3)
        Voronoi site colors
    """
    from skimage import filters
    import torch
    ## Load Images
    I_orig = I
    M = I.shape[0]
    N = I.shape[1]

    ## Sample points
    IGray = np.array(I_orig)
    if np.max(IGray) > 1:
        IGray = IGray/255
    if len(IGray.shape) > 2:
        IGray = 0.2125*IGray[:, :, 0] + 0.7154*IGray[:, :, 1] + 0.0721*IGray[:, :, 2]

    weights = filters.sobel(IGray)
    xy = rejection_sample_by_density(weights, n_points)
    xy = np.array(np.fliplr(xy))
    xy[:, 0] /= N
    xy[:, 1] /= M
    if do_weight_plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(weights, cmap='magma')
        plt.title("Weights")
        plt.subplot(122)
        plt.imshow(I)
        plt.scatter(xy[:, 0], xy[:, 1], s=1)
        plt.title("Samples")

    ## Setup pixel coordinates
    xpix, ypix = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, M), indexing='xy')
    xpix = xpix.flatten()
    ypix = ypix.flatten()
    X = np.array([xpix, ypix]).T

    ## Setup torch variables
    ICPU = np.reshape(I, (M*N, 3))/255
    I = torch.from_numpy(ICPU).to(device)

    xy = torch.from_numpy(xy)
    xy = torch.log(xy/(1-xy)) # Do inverse sigmoid
    xy = xy.to(device)
    xy = xy.requires_grad_()

    xpix = torch.from_numpy(xpix).to(device)
    ypix = torch.from_numpy(ypix).to(device)

    ## Do the regression!
    optimizer = torch.optim.Adam([xy], lr=lr)
    img_costs = []

    if plot_iter_interval > 0:
        plt.figure(figsize=(22, 10))

    plot_idx = 0
    for i in range(n_iters):
        tic = time.time()
        optimizer.zero_grad()
        
        J, rgb = splat_voronoi_image(torch.sigmoid(xy), X, xpix, ypix, I, ICPU, n_neighbs)
        img_cost = torch.sum((I-J)**2)
        loss = img_cost
        img_costs.append(img_cost.item())
        
        if plot_iter_interval > 0 and i % plot_iter_interval == 0:
            xydisp = torch.sigmoid(xy).detach().cpu().numpy()
            rgb = rgb.detach().cpu().numpy()
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
            plt.clf()
            plt.subplot2grid((6, 10), (0, 0), colspan=5, rowspan=5)
            plt.imshow(I_orig)
            plt.axis("off")
            plt.scatter(xydisp[:, 0]*N, xydisp[:, 1]*M, s=1)

            plt.subplot2grid((6, 10), (0, 5), colspan=5, rowspan=5)
            J = J.detach().cpu().numpy()
            J = np.reshape(J, (M, N, 3))
            J[J < 0] = 0
            J[J > 1] = 1
            plt.imshow(J)
            plt.axis("off")


            plt.subplot2grid((6, 10), (5, 0), colspan=10, rowspan=1)
            plt.plot(img_costs)
            plt.legend(["Img Costs"])
            plt.title("Img Cost = {:.3f}".format(img_cost.item()))
            plt.xlabel("Iteration")
            plt.ylabel("Squared Loss")

            plt.savefig("Voronoi{}.png".format(plot_idx), bbox_inches='tight')
            plot_idx += 1
        
        loss.backward()
        optimizer.step()
        
        if verbose:
            print("Iteration {}, Loss {:.3f}, Elapsed Time: {:.3f}".format(i, img_costs[-1], time.time()-tic))
    
    return J, xy, rgb