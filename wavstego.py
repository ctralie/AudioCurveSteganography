import numpy as np
from scipy.optimize import lsq_linear
from pywt import wavedec, waverec, dwt, idwt
from stego import *
import time


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

class WaveletCoeffs(StegoSolver):
    def __init__(self, x, target, win, fit_lam=1, wavtype='haar', wavlevel=7, coefflevel=1):
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
        wavtype: string
            Type of wavelet to use
        wavlevel: int
            Level of wavelets to go down to
        
        """
        StegoSolver.__init__(self, x, target)
        self.x_orig = np.array(x)
        self.target_orig = np.array(target)
        ## Step 1: Compute wavelets at all levels
        coeffs = wavedec(x, wavtype, level=wavlevel)
        self.wavlevel = wavlevel
        self.coefflevel = coefflevel
        self.coeffs = coeffs
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
            Mat = SlidingWindowSumMatrix(Y.size, win, fit_lam)
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
        self.reparam_targets(csm)
        
    def solve(self, verbose=0):
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
            self.pairs_sqr[coord] = lsq_linear(self.mats[coord], y, (0, np.inf), verbose=verbose)['x']
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
            Mat = SlidingWindowSumMatrix(Y.size, self.win, self.fit_lam)
            res = Mat.dot(Y)
            x = res[0:M]
            if normalize:
                x = (x-np.mean(x))/np.std(x)
            X[:, coord] = x
        return X