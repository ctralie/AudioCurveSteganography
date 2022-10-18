"""
General steganography utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.sparse.linalg import LinearOperator

def get_snr(x, y):
    """
    Compute the SNR in dB
    """
    N = min(x.size, y.size)
    x = x[0:N]
    y = y[0:N]
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


class SlidingWindowCentroidMatrix(LinearOperator):
    """
    Perform the effect of a sliding window average of a spectral centroid
    """
    def __init__(self, x_orig, win, denom_lam=1, fit_lam=1):
        """
        Parameters
        ----------
        x_orig: ndarray(K, N)
            Spectrogram subset
        win: int
            Window length to use
        denom_lam: float
            The weight to put on the denominator
        fit_lam: float
            The weight to put on the fit term
        """
        K = x_orig.shape[0] # Number of frequencies
        N = x_orig.shape[1] # Number of time points
        M = N-win+1
        self.K = K
        self.N = N
        self.shape = ((2*M+K*N, K*N))
        self.win = win
        self.denom_lam = denom_lam
        self.fit_lam = fit_lam
        self.dtype = float
        x_orig = np.sum(x_orig, axis=0)
        x_orig = np.concatenate(([0], x_orig))
        x_orig = np.cumsum(x_orig)
        self.denom = x_orig[win::]-x_orig[0:-win]
        self.mul_calls = 0
        self.rmul_calls = 0
            
    
    def _matvec(self, x_param):
        """
        y1 holds the weighted denominator
        y2 holds weighted sliding window sum
        y3 holds the fit to the original
        """
        K = self.K
        self.mul_calls += 1
        self.rmul_calls += 1
        x = np.zeros((K, x_param.size//K + 1))
        x[:, 1::] = np.reshape(x_param, (K, x_param.size//K))
        y1 = np.sum(x, axis=0)
        y1 = np.cumsum(y1)
        y1 = y1[self.win::]-y1[0:-self.win]
        y1 = (K/2)*self.denom_lam*y1/self.denom
        mul = np.arange(K)[:, None]
        x2 = np.sum(x*mul, axis=0)
        y2 = np.cumsum(x2)
        y2 = y2[self.win::]-y2[0:-self.win]
        y2 = y2/self.denom
        y3 = self.fit_lam*x_param.flatten()
        return np.concatenate((y1, y2, y3))
    
    def get_sliding_window_contrib(self, y, fac):
        """
        A helper function for the transpose
        """
        p = np.zeros(y.size+2*self.win-1)
        p[self.win:self.win+y.size] = y*fac
        p = np.cumsum(p)
        return p[self.win::] - p[0:-self.win]
    
    def _rmatvec(self, y):
        K = self.K
        N = self.N
        M = N-self.win+1
        y1 = y[0:M]
        y2 = y[M:2*M]
        x = np.zeros((K, N))
        p1 = self.get_sliding_window_contrib(y1, (K/2)*self.denom_lam/self.denom)
        x += p1[None, :]
        p = np.zeros((K, M+2*self.win-1))
        mul = np.arange(K)[:, None]
        p[:, self.win:self.win+M] = mul*y2[None, :]/self.denom[None, :]
        p = np.cumsum(p, axis=1)
        x += p[:, self.win::] - p[:, 0:-self.win]
        return x.flatten() + self.fit_lam*y[2*M::]


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


class StegoSolver:
    def __init__(self, x, target):
        """
        Parameters
        ----------
        x: ndarray(N pow of 2)
            Audio samples
        target: ndarray(M, dim)
            Target curve
        """
        self.x_orig = np.array(x)
        self.target_orig = np.array(target)
        ## Step 1: Compute wavelets at all levels
        self.dim = target.shape[1]
        self.targets = [] # Normalized target time series for each coordinate (dim total)


    def get_viterbi_path(self, csm):
        """
        Re-parameterize a bunch of targets to fit 

        Parameters
        ----------
        csm: ndarray(target_len, signal_len)
            Cross-similarity matrix between targets and signals
        """
        target_len = self.target_orig.shape[0]
        viterbi_K = 1
        finished = False
        path = []
        while not finished and viterbi_K < 20:
            pathk = viterbi_loop_trace(csm, viterbi_K)
            cost1 = np.sum(csm[pathk, np.arange(csm.shape[1])])
            path2 = viterbi_loop_trace(csm[:, ::-1], viterbi_K)
            path2.reverse()
            cost2 = np.sum(csm[path2, np.arange(csm.shape[1])])
            if cost2 < cost1:
                pathk = path2
            path_unwrap = np.unwrap(pathk, period=target_len)
            if np.abs(path_unwrap[0]-path_unwrap[-1]) >= target_len:
                finished = True
            else:
                viterbi_K += 1
            path = pathk
        print("viterbi_K = ", viterbi_K)
        plt.figure()
        plt.plot(path)
        plt.show()
        return path

    def reparam_targets(self, csm):
        """
        Re-parameterize a bunch of targets to fit 

        Parameters
        ----------
        csm: ndarray(target_len, signal_len)
            Cross-similarity matrix between targets and signals
        """
        path = self.get_viterbi_path(csm)
        self.targets = [t[path] for t in self.targets]
        
    def solve(self):
        """
        Perform linear least squares to perturb the wavelet coefficients
        to best match their targets
        """
        print("Error: calling solve on parent class")
    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting all perturbed transforms
        """
        print("Error: calling reconstruct_signal on parent class")
        return np.array([])

    def get_target(self, normalize=False):
        """
        Return the targets

        Parameters
        ----------
        normalize: boolean
            Whether to z-normalize each component
        
        Returns
        -------
        ndarray(N, k)
            Target components
        """
        ret = np.array(self.targets).T
        if normalize:
            ret = (ret - np.mean(ret, axis=0)[None, :])
            ret = ret/np.std(ret, axis=0)[None, :]
        return ret
    
    def get_signal(self, normalize=False):
        print("Error: calling get_signal on parent class")
        return np.array([])

    def plot(self, normalize=False):
        Y = self.get_target(normalize)
        Z = self.get_signal(normalize)
        res = 4
        plt.figure(figsize=(res*3, res*(self.dim+2)))
        for k in range(self.dim):
            plt.subplot2grid((self.dim*2, 2), (k, 0), colspan=2)
            plt.plot(Y[:, k])
            plt.plot(Z[:, k])
            plt.legend(["Target", "Signal"])
            plt.subplot2grid((self.dim*2, 2), (self.dim, k))
            plt.scatter(Y[:, k], Z[:, k], c=np.arange(Y.shape[0]), cmap='magma_r')
            plt.xlabel("Target")
            plt.ylabel("Signal")
        plt.subplot2grid((self.dim*2, 2), (self.dim+1, 0))
        plt.plot(Y[:, 0], Y[:, 1])
        plt.title("Target")
        plt.axis("equal")
        plt.subplot2grid((self.dim*2, 2), (self.dim+1, 1))
        plt.plot(Z[:, 0], Z[:, 1])
        mean, mx = self.get_distortion()
        L = Z[1::, :]-Z[0:-1, :]
        L = np.sqrt(np.sum(L**2, axis=1))
        L = np.sum(L)
        plt.title("Reconstructed (Mean Distortion = {:.3f}, Max Distortion = {:.3f})\nLength = {:.3f}".format(mean, mx, L))
        plt.axis("equal")
        plt.tight_layout()
    
    def get_distortion(self):
        """
        Compute the average and max geometric distortion between the original 
        target and the computed signal
        """
        Y = self.get_target(normalize=True)
        Z = self.get_signal(normalize=True)
        diff = np.sum((Y-Z)**2, axis=1)
        diff = np.sqrt(diff)
        return np.mean(diff), np.max(diff)


        """
        from scipy.spatial import KDTree
        Y = np.array(self.target_orig)
        Y -= np.mean(Y, axis=0)[None, :]
        Y /= np.std(Y, axis=0)[None, :]
        Z = self.get_signal(normalize=True)
        treeY = KDTree(Y)
        treeZ = KDTree(Z)
        dd1, _ = treeY.query(Z, k=1)
        dd2, _ = treeZ.query(Y, k=1)
        dd = np.concatenate((dd1.flatten(), dd2.flatten()))
        return np.mean(dd), np.max(dd)
        """