"""
General steganography utility functions
"""
import numpy as np
from numba import jit
from scipy.sparse.linalg import LinearOperator

def get_snr(x, y):
    """
    Compute the SNR in dB
    """
    power_sig = x**2
    power_noise = (y-x)**2
    snr = np.log10(np.mean(power_sig)) - np.log10(np.mean(power_noise))
    return 10*snr

class SlidingWindowSumMatrix(LinearOperator):
    """
    Perform the effect of a sliding window sum of an array
    """
    def __init__(self, N, win, fit_lam=1):
        """
        Parameters
        ----------
        N: int
            Length of signal to embed
        win: int
            Window length to use
        fit_lam: float
            The weight to put on the fit term
        """
        M = N-win+1
        self.N = N
        self.M = M
        self.shape = ((M+N, N))
        self.win = win
        self.fit_lam = fit_lam
        self.dtype = float
        self.mul_calls = 0
        self.rmul_calls = 0
        # Pre-allocated matrices that are used to help with cumulative sums
        self.xi = np.zeros(N+1)
        self.p = np.zeros(M+2*win-1)
    
    def _matvec(self, x_param):
        """
        y1 holds sliding window sum
        y2 holds weighted sliding window sum
        """
        self.mul_calls += 1
        self.rmul_calls += 1
        self.xi[1::] = x_param
        y1 = np.cumsum(self.xi)
        y1 = y1[self.win::]-y1[0:-self.win]
        return np.concatenate((y1, self.fit_lam*x_param.flatten()))
    
    def get_sliding_window_contrib(self, y, fac):
        """
        A helper function for the transpose
        """
        self.p[self.win:self.win+y.size] = y*fac
        p = np.cumsum(self.p)
        return p[self.win::] - p[0:-self.win]
    
    def _rmatvec(self, y):
        N = self.N
        M = self.M
        y1 = y[0:self.M]
        x = np.zeros(N)
        x += self.get_sliding_window_contrib(y1, 1)
        return x.flatten() + self.fit_lam*y[M::]

def get_window_energy(x, win, hop=1):
    """
    Return the sliding window squared energy of a signal

    Parameters
    ----------
    x: ndarray(N)
        Input signal
    win: int
        Window size
    hop: int
        Hop length between windows
    
    Returns
    -------
    ndarray(N-win+1)
        Windowed energy
    """
    eng = np.cumsum(np.concatenate(([0], x**2)))
    return eng[win::hop]-eng[0:-win:hop]

def get_normalized_target(x, target, min_freq=1, max_freq=2, stdev=2):
    """
    Normalize the target to the range of the centroid

    x: ndarray(N)
        Original Signal
    target: ndarray(T >= N)
        Target signal
    min_freq: int
        Minimum frequency index to use
    max_freq: int
        One beyond the maximum frequency index to use
    stdev: float
        How many standard deviations to fit
    """
    xrg = stdev*np.std(x)
    xmu = np.mean(x)
    xmin = max(min_freq, xmu-xrg)
    xmax = min(max_freq, xmu+xrg)
    target -= np.min(target)
    target /= np.max(target)
    return xmin + target*(xmax-xmin)

@jit(nopython=True)
def viterbi_loop_trace(csm, K):
    """
    Trace through a cyclic set of target states to best match the L1
    norm between target states and time points

    Parameters
    ----------
    csm: ndarray(M, N)
        L1 Cross-similarity between M target states and N time points
    K: int
        Maximum jump interval between states
    
    Returns
    -------
    list(N)
        State indices of best fit cyclic path
    """
    M = csm.shape[0]
    N = csm.shape[1]
    S = np.zeros((M, N))
    S[:, 0] = csm[:, 0]
    B = np.zeros((M, N)) # Backpointers
    for j in range(1, N):
        for i in range(M):
            idxmin = i-K
            valmin = np.inf
            for k in range(i-K, i):
                k = k%M
                if S[k, j-1] < valmin:
                    valmin = S[k, j-1]
                    idxmin = k
            S[i, j] = valmin + csm[i, j]
            B[i, j] = idxmin
    j = N-1
    i = np.argmin(S[:, -1])
    path = []
    while j > 0:
        path.append(i)
        i = int(B[i, j])
        j -= 1
    path.append(i)
    path.reverse()
    return path

def get_best_target(X, Y, K):
    """
    Return the best path through a target, traversing it in either direction

    Parameters
    ----------
    X: ndarray(M, 2)
        Target states
    Y: ndarray(N, 2)
        Time points
    K: int
        Maximum jump interval between states
    """
    costfn = lambda Z: np.sum(np.abs(Z-Y))

    min_path = np.arange(Y.shape[0]) % X.shape[0]
    min_cost = costfn(X[min_path, :])
    print("Default cost", min_cost)

    # Try default orientation
    csm = np.abs(X[:, 0][:, None] - Y[:, 0][None, :])
    csm += np.abs(X[:, 1][:, None] - Y[:, 1][None, :])
    path = viterbi_loop_trace(csm, K)
    path = np.array(path, dtype=int)
    cost = costfn(X[path, :])
    print("Cost", cost)
    if cost < min_cost:
        min_cost = cost
        min_path = path
    
    # Try reverse orientation
    csm = np.abs(X[:, 0][:, None] - Y[::-1, 0][None, :])
    csm += np.abs(X[:, 1][:, None] - Y[::-1, 1][None, :])
    path = viterbi_loop_trace(csm, K)
    path = X.shape[0]-1-np.array(path, dtype=int)
    cost = costfn(X[path, :])
    print("Reverse cost", cost)
    if cost < min_cost:
        print("Reverse wins!")
        min_cost = cost
        min_path = path

    return min_path, min_cost

def make_voronoi_image(coords, phases):
    from scipy.spatial import KDTree
    phases = np.abs(phases)
    i1, j1 = np.min(coords, axis=0)
    i2, j2 = np.max(coords, axis=0)
    I, J = np.meshgrid(np.arange(i1, i2+1), np.arange(j1, j2+1), indexing='ij')
    shape = I.shape
    I = I.flatten()
    J = J.flatten()
    tree = KDTree(coords)
    _, idx = tree.query(np.array([I, J]).T)
    return np.reshape(phases[idx], shape)

@jit(nopython=True)
def get_maxes(S, max_freq, time_win, freq_win):
    ret = []
    M, N = S.shape
    for i in range(max_freq):
        for j in range(N):
            constraint = True
            ni = max(0, i-freq_win)
            while constraint and ni < min(max_freq, i+freq_win+1):
                nj = max(0, j-time_win)
                while constraint and nj < min(N, j+time_win+1):
                    if ni != i or nj != j:
                        if S[ni, nj] > S[i, j]:
                            constraint = False
                    nj += 1
                ni += 1
            if constraint:
                ret.append([i, j])
    return ret