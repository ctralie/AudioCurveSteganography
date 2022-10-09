from tkinter import LabelFrame
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import lsq_linear
from pywt import wavedec, waverec, dwt, idwt
from stego import *
import time


class SlidingWindowCentroidMatrix(LinearOperator):
    """
    Perform the effect of a sliding window average between two
    parallel arrays
    """
    def __init__(self, x_orig, win, lam=1):
        """
        Parameters
        ----------
        x_orig: ndarray(2, N)
            The two signals
        win: int
            Window length to use
        lam: float
            The weight to put on fit to sliding window sum
        """
        N = x_orig.shape[1]
        M = N-win+1
        self.N = N
        self.shape = ((2*M, 2*N))
        self.win = win
        self.lam = lam
        self.dtype = float
        x_orig = np.sum(x_orig, axis=0)
        x_orig = np.concatenate(([0], x_orig))
        x_orig = np.cumsum(x_orig)
        self.denom = x_orig[win::]-x_orig[0:-win]
            
    
    def _matvec(self, x):
        """
        y1 holds sliding window sum
        y2 holds weighted sliding window sum
        """
        x = np.reshape(x, (2, x.size//2))
        x = np.concatenate(([[0], [0]], x), axis=1)
        y1 = np.sum(x, axis=0)
        y1 = np.cumsum(y1)
        y1 = y1[self.win::]-y1[0:-self.win]
        y1 = 1.5*self.lam*y1/self.denom
        mul = np.array([[1], [2]])
        x2 = np.sum(x*mul, axis=0)
        y2 = np.cumsum(x2)
        y2 = y2[self.win::]-y2[0:-self.win]
        y2 = y2/self.denom
        return np.concatenate((y1, y2))
    
    def get_sliding_window_contrib(self, y, fac):
        """
        A helper function for the transpose
        """
        p = np.zeros(y.size+2*self.win-1)
        p[self.win:self.win+y.size] = y*fac
        p = np.cumsum(p)
        return p[self.win::] - p[0:-self.win]
        
    
    def _rmatvec(self, y):
        N = self.N
        y1 = y[0:y.size//2]
        y2 = y[y.size//2::]
        x = np.zeros(2*N)
        # for i, yi in enumerate(y[0:y.size//2]):
        #     x[0, i:i+self.win] += 1.5*self.lam*yi/self.denom[i]
        #     x[1, i:i+self.win] += 1.5*self.lam*yi/self.denom[i]
        p1 = self.get_sliding_window_contrib(y1, 1.5*self.lam/self.denom)
        x[0:N] += p1
        x[N::] += p1
        #for i, yi in enumerate(y[y.size//2::]):
        #    x[0, i:i+self.win] += yi/self.denom[i]
        #    x[1, i:i+self.win] += 2*yi/self.denom[i]
        x[0:N] += self.get_sliding_window_contrib(y2, 1/self.denom)
        x[N::] += self.get_sliding_window_contrib(y2, 2/self.denom)
        return x

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
    def __init__(self, x, target, win, lam=1, k=2, viterbi_K=4, wavtype='haar', wavlevel=4):
        """
        Parameters
        ----------
        x: ndarray(N pow of 2)
            Audio samples
        target: ndarray(M, 2)
            Target curve
        win: int
            Window length to use
        lam: float
            The weight to put on fit to sliding window sum
        k: int
            Index of first wavelet to split up
        viterbi_K: int
            Maximum jump allowed during re-parameterization
        wavtype: string
            Type of wavelet to use
        wavlevel: int
            Level of wavelets to go down to
        
        """
        ## Step 1: Compute wavelets at all levels
        coeffs = wavedec(x, wavtype, level=wavlevel)
        coeffs = [[c] for c in coeffs[::-1]]
        self.coeffs = coeffs
        self.k = k
        self.wavtype = wavtype
        self.win = win
        self.lam = lam
        
        divamt = 1+int(np.round(np.log2(coeffs[k][0].size/(target.shape[0]+win))))
        coeffs[k] = subdivide_wavlevel_dec(coeffs[k][0], divamt, wavtype)
        coeffs[k+1] = subdivide_wavlevel_dec(coeffs[k+1][0], divamt-1, wavtype)
        print("coeffs size", coeffs[k][0].size)
        
        ## Step 2: Setup all aspects of sliding windows
        self.signs = [np.sign(x) for x in coeffs[k] + coeffs[k+1]] # Signs of wavelet coefficients before squaring
        self.pairs_sqr = [p**2 for p in coeffs[k] + coeffs[k+1]] # Squared wavelet coefficients
        self.mats = [] # Matrices to do sliding window averaging transforms on each pair
        self.targets = [] # Normalized target time series for each pair
        self.coords = [] # Coordinates of each pair
        csm = np.array([]) # Cross-similarity matrix for aligned targets
        for i in range(0, len(self.signs), 2):
            Y = np.array([self.pairs_sqr[i], self.pairs_sqr[i+1]])
            Mat = SlidingWindowCentroidMatrix(Y, win, lam)
            self.mats.append(Mat)
            res = Mat.dot(Y.flatten())
            res = res[res.size//2::]
            targeti = []
            if i < len(self.signs)//2:
                # First half of pairs are the x coordinate
                targeti = np.array(target[:, 0])
                self.coords.append(0)
            else:
                # Second half of pairs are the y coordinate
                targeti = np.array(target[:, 1])
                self.coords.append(1)
            targeti = get_normalized_target(res, targeti, 1, 2)
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
        M = len(self.targets[0])
        for i in range(0, len(self.targets)):
            tic = time.time()
            print("Computing target {} of {}...\n".format(i+1, len(self.targets)))
            y = 1.5*self.lam*np.ones(M*2)
            y[M::] = self.targets[i]
            mx = 1.1*max(np.max(self.pairs_sqr[i*2]), np.max(self.pairs_sqr[i*2+1]))
            res = lsq_linear(self.mats[i], y, (0, mx))
            self.pairs_sqr[i*2] = res[0:res.size//2]
            self.pairs_sqr[i*2+1] = res[res.size//2::]
            print("Elapsed time: {:.3f}".format(time.time()-tic))


    def get_avg(self, Xs, normalize=False):
        """
        Return the average of elements in an array

        Parameters
        ----------
        Xs: list of ndarray(N)
            Arrays to average
        normalize: bool
            Whether to z-normalize each array

        Returns
        -------
        averaged
        """
        X = np.zeros_like(Xs[0])
        for Xi in Xs:
            if normalize:
                Xi = (Xi-np.mean(Xi))/np.std(Xi)
            X += Xi
        return X / len(Xs)

    def get_target_avg(self, normalize=False):
        """
        Compute the averaged of the z-normalized target components

        Returns
        -------
        ndarray(M, 2)
            The normalized target
        """
        M = self.targets[0].size
        Y = np.zeros((M, 2))
        for k in range(2):
            Y[:, k] = self.get_avg([t for i, t in zip(self.coords, self.targets) if i == k], normalize)
        return Y
    
    def get_signal_avg(self, normalize=False):
        """
        Compute the average of the sliding window centroids

        Returns
        -------
        ndarray(M, 2)
            Average of the sliding window centroids
        """

        k = len(np.unique(self.coords))
        M = self.targets[0].size
        X = np.zeros((M, k))
        counts = np.zeros(k)
        for i in range(0, len(self.pairs_sqr), 2):
            Y = np.array([self.pairs_sqr[i], self.pairs_sqr[i+1]])
            Mat = self.mats[i//2]
            res = Mat.dot(Y.flatten())
            res = res[res.size//2::]
            k = self.coords[i//2]
            if normalize:
                res = (res-np.mean(res))/np.std(res)
            X[:, k] += res
            counts[k] += 1
        return X/counts[None, :]
    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting all wavelet transforms
        """
        coeffs = self.coeffs.copy()
        coeffs_new = []
        for s, p in zip(self.signs, self.pairs_sqr):
            p[p < 0] = 0
            coeffs_new.append(np.sqrt(p)*s)
        coeffs1 = coeffs_new[0:len(self.coeffs[self.k])]
        coeffs2 = coeffs_new[len(self.coeffs[self.k])::]
        coeffs[self.k] = [subdivide_wavelet_rec(coeffs1, self.wavtype)]
        coeffs[self.k+1] = [subdivide_wavelet_rec(coeffs2, self.wavtype)]
        return waverec([c[0] for c in coeffs[::-1]], self.wavtype)
    
    def plot(self, normalize=False):
        Y = self.get_target_avg(normalize)
        Z = self.get_signal_avg(normalize)
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(Y[:, 0])
        plt.plot(Z[:, 0])
        plt.title("X Coordinate")
        plt.legend(["Target", "Signal"])
        plt.subplot(212)
        plt.plot(Y[:, 1])
        plt.plot(Z[:, 1])
        plt.title("Y Coordinate")
        plt.legend(["Target", "Signal"])

        plt.figure(figsize=(6, 6))
        plt.plot(Y[:, 0], Y[:, 1])
        plt.plot(Z[:, 0], Z[:, 1])
        plt.axis("equal")