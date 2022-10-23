import numpy as np
import matplotlib.pyplot as plt
from numba import jit

DTW_UP = 0
DTW_LEFT = 1
DTW_DIAG = 2

def get_csm(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    Parameters
    ----------
    X: ndarray(M, d)
        A point cloud with M points in d dimensions
    Y: ndarray(N, d)
        A point cloud with N points in d dimensions
    Returns
    -------
    D: ndarray(M, N)
        An MxN Euclidean cross-similarity matrix
    """
    if len(X.shape) == 1:
        X = X[:, None]
    if len(Y.shape) == 1:
        Y = Y[:, None]
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def circ_shift(X, k):
    """
    Do a left circular shift by k places
    """
    assert(k < X.shape[0])
    return np.concatenate((X[k::, :], X[0:k, :]))

@jit(nopython=True)
def dtw(csm):
    """
    Perform dynamic time warping between two time series

    Parameters
    ----------
    csm: ndarray(M, N)
        Cross-similarity matrix between X (of length M) and Y (of length N)
    
    Returns
    -------
    S: ndarray(M, N)
        Dynamic programming matrix
    P: ndarray(M, N, dtype=int)
        Backpointer matrix
    """
    M = csm.shape[0]
    N = csm.shape[1]
    S = np.zeros((M, N)) # The dynamic programming matrix
    P = np.zeros((M, N)) # The backtracing matrix
    S[0, 0] = csm[0, 0]
    for i in range(M):
        for j in range(N):
            if i != 0 or j != 0:
                S[i,j] = csm[i, j]
                mn = np.inf
                mnidx = -1
                # UP
                if i > 0:
                    if S[i-1, j] < mn:
                        mn = S[i-1, j]
                        mnidx = DTW_UP
                # LEFT
                if j > 0:
                    if S[i, j-1] < mn:
                        mn = S[i, j-1]
                        mnidx = DTW_LEFT
                # DIAG
                if i > 0 and j > 0:
                    if S[i-1, j-1] < mn:
                        mn = S[i-1,j-1]
                        mnidx = DTW_DIAG
                S[i, j] += mn
                P[i, j] = mnidx
    return S, P

def dtw_backtrace(P, i, j, verbose=False):
    """
    Backtrace a dynamic programming matrix starting at [i, j]
    
    Parameters
    ----------
    P: ndarray(M, N, dtype=int)
        Backpointer matrix
    i: int
        Row to start backtracing
    j: int
        Column to start backtracing
    
    Returns
    -------
    ndarray(K, 2)
        Warping path
    """
    path = [[i, j]]
    dirs = {}
    dirs[DTW_UP] = [-1, 0]
    dirs[DTW_LEFT] =[0, -1]
    dirs[DTW_DIAG] = [-1, -1]
    while i > 0 or j > 0:
        dir = dirs[int(P[i][j])]
        i += dir[0]
        j += dir[1]
        path.append([i, j])
        if verbose:
            print(i, j)
    path.reverse()
    return np.array(path, dtype=int)

def dtw_cyclic_brute(csm):
    """
    Perform cyclic DTW between two time series, based on their cross-similarity matrix
    This is a brute force version that costs O(MN^2)

    Parameters
    ----------
    csm: ndarray(M, N)
        Cross-similarity matrix between X (of length M) and Y (of length N)
    
    Returns
    -------
    idx: int
        Index of best shift
    cost: float
        Cost of best shifted DTW
    path: ndarray(K, 2)
        Warping path corresponding to best shifted DTW
    """
    M = csm.shape[0]
    N = csm.shape[1]
    D = np.concatenate((csm, csm), axis=1)

    min_idx = -1
    min_cost = np.inf
    min_path = np.array([])
    for k in range(N):
        S, P = dtw(D[:, k:k+N+1])
        path = np.array([])
        cost = np.inf
        if S[-1, -1] < S[-1, -2]:
            cost = S[-1, -1]
            path = dtw_backtrace(P, M-1, N)
        else:
            cost = S[-1, -2]
            path = dtw_backtrace(P, M-1, N-1)
        path[:, 1] = (path[:, 1] + k) % N
        if cost < min_cost:
            min_idx = k
            min_cost = cost
            min_path = np.array(path, dtype=int)
    return min_idx, min_cost, min_path

@jit(nopython=True)
def dtw_constrained(csm, S, P, jstart, left_path, right_path):
    """
    Perform dynamic time warping between two time series, and restrict the
    solution to lie in between two warping paths

    Parameters
    ----------
    csm: ndarray(M, N)
        Cross-similarity matrix between X (of length M) and Y (of length N)
    S: ndarray(M, N)
        Pre-allocated dynamic programming matrix, by reference
    P: ndarray(M, N)
        Pre-allocated backpointer matrix, by reference
    jstart: int
        Start column of warping path
    left_path: ndarray(K1, 2)
        Left warping path
    right_path: ndarray(K2, 2)
        Right warping path
    """
    M = csm.shape[0]
    N = csm.shape[1]//2
    ## Step 1: Determine the column constraints
    j1 = -1*np.ones(M)
    j2 = -1*np.ones(M)
    for [i, j] in left_path:
        if j1[i] == -1:
            j1[i] = max(j, jstart)
    for [i, j] in right_path:
        j2[i] = min(j, jstart+N)
    
    ## Step 3: Do constrained dynamic time warping
    S[0, jstart] = csm[0, jstart]
    for i in range(M):
        for j in range(int(j1[i]), int(j2[i]+1)):
            if i != 0 or j != jstart:
                S[i,j] = csm[i, j]
                mn = np.inf
                mnidx = -1
                # UP
                if i > 0 and j >= j1[i-1] and j <= j2[i-1]:
                    if S[i-1, j] < mn:
                        mn = S[i-1, j]
                        mnidx = DTW_UP
                # LEFT
                if j-1 >= j1[i] and j-1 <= j2[i]:
                    if S[i, j-1] < mn:
                        mn = S[i, j-1]
                        mnidx = DTW_LEFT
                # DIAG
                if i > 0 and j-1 >= j1[i-1] and j-1 <= j2[i-1]:
                    if S[i-1,j-1] < mn:
                        mn = S[i-1,j-1]
                        mnidx = DTW_DIAG
                S[i, j] += mn
                P[i, j] = mnidx


def dtw_cyclic_rec(D, S, P, l, left_path, r, right_path, solutions):
    """
    Recursively perform cyclic DTW between two time series, based on their cross-similarity matrix
    
    Parameters
    ----------
    D: ndarray(M, 2N)
        Twice repeated cross-similarity matrix between X (of length M) and Y (of length N)
    S: ndarray(M, 2N)
        Staging area for dtw
    P: ndarray(M, 2N)
        Staging area for backpointers
    l: int
        Index of left warping path
    left_path: ndarray(Kl, 2)
        Left warping path
    r: int
        Index of right warping path
    right_path: ndarray(Kr, 2)
        Right warping path
    solutions: dictionary of index->(cost, path)
        All costs and warping paths
    """
    M = D.shape[0]
    N = D.shape[1]//2
    if r > l+1:
        k = (r+l)//2
        jstart = k
        jend = jstart + N
        S = -1*np.ones_like(D)
        P = -1*np.ones(D.shape, dtype=int)
        dtw_constrained(D, S, P, jstart, left_path, right_path)
        mid_cost = np.inf
        if S[-1, jend] < S[-1, jend-1]:
            mid_path = dtw_backtrace(P[:, k:k+N+1], M-1, N)
            mid_cost = S[-1, jend]
        else:
            mid_path = dtw_backtrace(P[:, k:k+N+1], M-1, N-1)
            mid_cost = S[-1, jend-1]
        mid_path[:, 1] += k
        solutions[k] = (mid_cost, mid_path)
        # Recursive calls
        dtw_cyclic_rec(D, S, P, l, left_path, k, mid_path, solutions)
        dtw_cyclic_rec(D, S, P, k, mid_path, r, right_path, solutions)

def dtw_cyclic(csm):
    """
    Recursively perform cyclic DTW between two time series, based on their cross-similarity matrix
    
    Parameters
    ----------
    csm: ndarray(M, 2N)
        Cross-similarity matrix between X (of length M) and Y (of length N)

    Returns
    -------
    idx: int
        Index of best shift
    cost: float
        Cost of best shifted DTW
    path: ndarray(K, 2)
        Warping path corresponding to best shifted DTW
    """
    M = csm.shape[0]
    N = csm.shape[1]
    D = np.concatenate((csm, csm), axis=1)
    
    ## Step 1: Compute left path
    S, P = dtw(D[:, 0:N+1])
    if S[-1, -1] < S[-1, -2]:
        left_path = dtw_backtrace(P, M-1, N)
    else:
        left_path = dtw_backtrace(P, M-1, N-1)
    left_cost = min(S[-1, -1], S[-1, -2])
    
    ## Step 2: Compute right path
    S, P = dtw(D[:, N-1::])
    if S[-1, -1] < S[-1, -2]:
        right_path = dtw_backtrace(P, M-1, N)
    else:
        right_path = dtw_backtrace(P, M-1, N-1)
    right_cost = min(S[-1, -1], S[-1, -2])
    right_path[:, 1] = right_path[:, 1] + N-1
    
    ## Step 3: Kick off recursion
    S = np.zeros_like(csm)
    P = np.zeros(S.shape, dtype=int)
    solutions = {0:(left_cost, left_path), N-1:(right_cost, right_path)}
    dtw_cyclic_rec(D, S, P, 0, left_path, N-1, right_path, solutions)
    
    ## Step 4: Extract best cost path
    costs = np.array([solutions[i][0] for i in range(N)])
    idx = np.argmin(costs)
    path = solutions[idx][1]
    path[:, 1] %= N
    return idx, costs[idx], path