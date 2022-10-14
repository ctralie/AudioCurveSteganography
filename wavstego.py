from tkinter import LabelFrame
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import lsq_linear
from pywt import wavedec, waverec, dwt, idwt
import cvxpy as cp
from stego import *
import time

class SlidingWindowCentroidMatrix(LinearOperator):
    """
    Perform the effect of a sliding window average between two
    parallel arrays
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


def subdivide_wavlevel_dec(x, max_depth, wavtype, depth=1):
    res = []
    if depth == max_depth:
        res = [x]
    else:
        cA, cD = dwt(x, wavtype)
        res = subdivide_wavlevel_dec(cA, max_depth, wavtype, depth+1)
        res += subdivide_wavlevel_dec(cD, max_depth, wavtype, depth+1)
    return res

def subdivide_wavelet_rec(coeffs, wavtype):
    cA = None
    cD = None
    if len(coeffs) == 2:
        [cA, cD] = coeffs
    else:
        N = len(coeffs)//2
        cA = subdivide_wavelet_rec(coeffs[0:N], wavtype)
        cD = subdivide_wavelet_rec(coeffs[N:], wavtype)
    return idwt(cA, cD, wavtype)

class WaveletCoeffs:
    def __init__(self, x, target, win, fit_lam=1, k=2, viterbi_K=4, wavtype='haar', wavlevel=7, coefflevel=1):
        """
        Parameters
        ----------
        x: ndarray(N pow of 2)
            Audio samples
        target: ndarray(M, dim)
            Target curve
        win: int
            Window length to use
        fit_lam: float
            Weight to put into the fit
        viterbi_K: int
            Maximum jump allowed during re-parameterization
        wavtype: string
            Type of wavelet to use
        wavlevel: int
            Level of wavelets to go down to
        
        """
        ## Step 1: Compute wavelets at all levels
        coeffs = wavedec(x, wavtype, level=wavlevel)
        self.wavlevel = wavlevel
        self.coefflevel = coefflevel
        self.coeffs = coeffs
        self.dim = target.shape[1]
        self.wavtype = wavtype
        self.win = win
        self.fit_lam = fit_lam
        
        coeffs_mod = wavedec(coeffs[coefflevel], wavtype, level=1)

        ## Step 2: Setup all aspects of sliding windows
        self.signs = [np.sign(x) for x in coeffs_mod] # Signs of wavelet coefficients before squaring
        self.pairs_sqr = [p**2 for p in coeffs_mod] # Squared wavelet coefficients
        self.mats = [] # Matrices to do sliding window averaging transforms on each group of coefficients (dim total)
        self.targets = [] # Normalized target time series for each coordinate (dim total)
        self.coords = [] # Coordinate indices of each group of coefficients
        csm = np.array([]) # Cross-similarity matrix for aligned targets
        K = len(self.signs)//self.dim
        for coord in range(self.dim):
            self.coords += [coord]*K
            Y = self.pairs_sqr[coord]
            M = Y.size-win+1
            Mat = SlidingWindowCentroidMatrix(Y.size, win, fit_lam)
            self.mats.append(Mat)
            res = Mat.dot(Y)
            res = res[0:M]
            targeti = []
            targeti = np.array(target[:, coord])
            targeti = get_normalized_target(res, targeti, 0, np.inf)
            self.targets.append(targeti)
            csmi = np.abs(targeti[:, None] - res[None, :])
            if csm.size == 0:
                csm = csmi
            else:
                csm += csmi
        self.csm = csm
        self.coords = np.array(self.coords, dtype=int)

        ## Step 3: Re-parameterize targets
        path1 = viterbi_loop_trace(csm, viterbi_K)
        cost1 = np.sum(csm[path1, np.arange(csm.shape[1])])
        path = path1
        path2 = viterbi_loop_trace(csm[:, ::-1], viterbi_K)
        path2.reverse()
        cost2 = np.sum(csm[path2, np.arange(csm.shape[1])])
        if cost2 < cost1:
            path = path2
        self.targets = [t[path] for t in self.targets]
        
    def solve(self):
        """
        Perform linear least squares to perturb the wavelet coefficients
        to best match their targets
        """
        M = self.targets[0].size
        for coord in range(self.dim):
            tic = time.time()
            print("Computing target coordinate {} of {}...\n".format(coord+1, self.dim))
            Y = self.pairs_sqr[coord]
            y = np.ones(M+Y.size)
            y[0:M] = self.targets[coord]
            y[M::] = self.fit_lam*Y
            self.pairs_sqr[coord] = lsq_linear(self.mats[coord], y, (0, np.inf), verbose=2)['x']
            print("Elapsed time: {:.3f}".format(time.time()-tic))
    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting all wavelet transforms
        """
        coeffs = self.coeffs.copy()
        coeffs_mod = []
        for s, p in zip(self.signs, self.pairs_sqr):
            print(np.sum(p < 0)/p.size)
            p = np.array(p)
            p[p < 0] = 0
            coeffs_mod.append(s*np.sqrt(p))
        coeffs[self.coefflevel] = waverec(coeffs_mod, self.wavtype)
        y = waverec(coeffs, self.wavtype)
        return y

    def get_target(self, normalize=False):
        ret = np.array(self.targets).T
        if normalize:
            ret = (ret - np.mean(ret, axis=0)[None, :])
            ret = ret/np.std(ret, axis=0)[None, :]
        return ret

    def get_signal(self, normalize=False):
        """
        Compute and z-normalize the sliding window centroids

        Returns
        -------
        ndarray(M, dim)
            Average of the sliding window centroids
        """
        M = self.targets[0].size
        X = np.zeros((M, self.dim))
        for coord in range(self.dim):
            Y = self.pairs_sqr[coord]
            Mat = SlidingWindowCentroidMatrix(Y.size, self.win, self.fit_lam)
            res = Mat.dot(Y)
            x = res[0:M]
            if normalize:
                x = (x-np.mean(x))/np.std(x)
            X[:, coord] = x
        return X

    def plot(self, normalize=False):
        Y = self.get_target(normalize)
        Z = self.get_signal(normalize)
        res = 4
        plt.figure(figsize=(res*3, res*self.dim))
        for k in range(self.dim):
            plt.subplot(self.dim, 1, k+1)
            plt.plot(Y[:, k])
            plt.plot(Z[:, k])
            plt.legend(["Target", "Signal"])

        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(Z[:, 0], Z[:, 1], c='C1')
        plt.subplot(122)
        plt.plot(Y[:, 0], Y[:, 1])
        plt.plot(Z[:, 0], Z[:, 1])
        plt.axis("equal")