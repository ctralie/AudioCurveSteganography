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
    def __init__(self, x_orig, win, denom_lam=1, fit_lam=1):
        """
        Parameters
        ----------
        x_orig: ndarray(2K, N)
            Two sets of signals between which to compute the average
        win: int
            Window length to use
        denom_lam: float
            The weight to put on the denominator
        fit_lam: float
            The weight to put on the fit term
        """
        K = x_orig.shape[0]
        N = x_orig.shape[1]
        M = N-win+1
        self.K = K
        self.N = N
        self.M = M
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
        # Pre-allocated matrices that are used to help with cumulative sums
        self.xi = np.zeros((K, N+1))
        self.p = np.zeros(M+2*win-1)
            
    
    def _matvec(self, x_param):
        """
        y1 holds sliding window sum
        y2 holds weighted sliding window sum
        """
        K = self.K
        self.mul_calls += 1
        self.rmul_calls += 1
        self.xi[:, 1::] = np.reshape(x_param, (K, x_param.size//K))
        y1 = np.sum(self.xi, axis=0)
        y1 = np.cumsum(y1)
        y1 = y1[self.win::]-y1[0:-self.win]
        y1 = 1.5*self.denom_lam*y1/self.denom
        x2 = np.sum(self.xi[K//2::, :], axis=0)
        y2 = np.cumsum(x2)
        y2 = y2[self.win::]-y2[0:-self.win]
        y2 = y2/self.denom
        return np.concatenate((y1, y2, self.fit_lam*x_param.flatten()))
    
    def get_sliding_window_contrib(self, y, fac):
        """
        A helper function for the transpose
        """
        self.p[self.win:self.win+y.size] = y*fac
        p = np.cumsum(self.p)
        return p[self.win::] - p[0:-self.win]
        
    
    def _rmatvec(self, y):
        K = self.K
        N = self.N
        M = self.M
        y1 = y[0:self.M]
        y2 = y[self.M:self.M*2]
        x = np.zeros((K, N))
        p1 = self.get_sliding_window_contrib(y1, 1.5*self.denom_lam/self.denom)
        x += p1[None, :]
        x[K//2::, :] += self.get_sliding_window_contrib(y2, 1/self.denom)[None, :]
        return x.flatten() + self.fit_lam*y[2*M::]


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
    def __init__(self, x, target, win, fit_lam=1, denom_lam=1, k=2, viterbi_K=4, wavtype='haar', wavlevel=4):
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
        denom_lam: float
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
        self.dim = target.shape[1]
        self.wavtype = wavtype
        self.win = win
        self.denom_lam = denom_lam
        self.fit_lam = fit_lam
        
        divamt = 1+int(np.round(np.log2(coeffs[k][0].size/(target.shape[0]+win))))
        coeffs[k] = subdivide_wavlevel_dec(coeffs[k][0], divamt, wavtype)
        coeffs[k+1] = subdivide_wavlevel_dec(coeffs[k+1][0], divamt-1, wavtype)
        print("coeffs size", coeffs[k][0].size)
        
        ## Step 2: Setup all aspects of sliding windows
        self.signs = [np.sign(x) for x in coeffs[k] + coeffs[k+1]] # Signs of wavelet coefficients before squaring
        self.pairs_sqr = [p**2 for p in coeffs[k] + coeffs[k+1]] # Squared wavelet coefficients
        print("len(self.pairs_sqr)", len(self.pairs_sqr))
        self.mats = [] # Matrices to do sliding window averaging transforms on each group of coefficients (dim total)
        self.targets = [] # Normalized target time series for each coordinate (dim total)
        self.coords = [] # Coordinate indices of each group of coefficients
        csm = np.array([]) # Cross-similarity matrix for aligned targets
        K = len(self.signs)//self.dim
        for coord in range(self.dim):
            self.coords += [coord]*K
            Y = np.array(self.pairs_sqr[coord*K:(coord+1)*K])
            M = Y.shape[1]-win+1
            Mat = SlidingWindowCentroidMatrix(Y, win, denom_lam, fit_lam)
            self.mats.append(Mat)
            res = Mat.dot(Y.flatten())
            res = res[M:M*2]
            targeti = []
            targeti = np.array(target[:, coord])
            targeti = get_normalized_target(res, targeti, 0, 1)
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
        K = len(self.pairs_sqr)//self.dim
        for coord in range(self.dim):
            tic = time.time()
            print("Computing target coordinate {} of {}...\n".format(coord+1, self.dim))
            Y = np.array(self.pairs_sqr[coord*K:(coord+1)*K])
            shape = Y.shape
            Y = Y.flatten()
            y = 1.5*self.denom_lam*np.ones(M*2+Y.size)
            y[M:M*2] = self.targets[coord]
            y[2*M::] = self.fit_lam*Y

            res = lsq_linear(self.mats[coord], y, (0, np.inf), verbose=2)['x']
            """
            x = cp.Variable(self.pairs_sqr[0].size)
            objective = cp.Minimize(cp.sum_squares(self.mats[i] @ x - y))
            constraints = [0 <= x, x <= mx]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            res = np.array(x.value)
            """
            res = np.reshape(res, shape)
            for k in range(res.shape[0]):
                self.pairs_sqr[coord*K+k] = res[k, :]
            print("Elapsed time: {:.3f}".format(time.time()-tic))
    
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
        K = len(self.pairs_sqr)//self.dim
        X = np.zeros((M, self.dim))
        for coord in range(self.dim):
            Y = np.array(self.pairs_sqr[coord*K:(coord+1)*K])
            Mat = SlidingWindowCentroidMatrix(Y, self.win, self.denom_lam)
            self.mats.append(Mat)
            res = Mat.dot(Y.flatten())
            x = res[M:M*2]
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