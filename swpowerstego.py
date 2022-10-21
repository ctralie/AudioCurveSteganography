import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from pywt import wavedec, waverec, dwt, idwt
from stego import *
import time

class StegoWindowedPower(StegoSolver):
    """
    A class for doing sliding window power on some nonnegative coefficients
    """
    def __init__(self, target, coeffs, win, fit_lam=1, q=-1):
        """
        Parameters
        ----------
        target: ndarray(M, dim)
            Target curve
        coeffs: list of dim ndarray(N)
            Coefficients to perturb.  Must be nonnegative
        win: int
            Window length to use
        fit_lam: float
            Weight to put into the fit
        q: float in [0, 1]
            Quantile in which to keep magnitudes.
            If -1, go up to infinity
        """
        self.win = win
        self.fit_lam = fit_lam
        self.q = q

        ## Step 2: Setup all aspects of sliding windows
        self.coeffs_orig = [np.array(c) for c in coeffs]
        coeffs = [np.array(c) for c in coeffs]
        self.coeffs = coeffs
        N = len(self.coeffs[0])
        Mat = SlidingWindowSumMatrix(N, win, fit_lam) # Matrix to do sliding window averaging transforms on each group of coefficients (dim total)
        self.Mat = Mat
        self.targets = [] # Normalized target time series for each coordinate (dim total)
        csm = np.array([]) # Cross-similarity matrix for aligned targets
        for coord in range(self.dim):
            Y = self.coeffs[coord]
            M = Y.size-win+1
            res = Mat.dot(Y)
            res = res[0:M]
            targeti = np.array(target[:, coord])
            targeti = get_normalized_target(res, targeti, 0, np.inf)
            self.targets.append(targeti)
            csmi = np.abs(targeti[:, None] - res[None, :])
            if csm.size == 0:
                csm = csmi
            else:
                csm += csmi
        self.csm = csm
        self.path =self.reparam_targets(csm)
        
    def solve(self, verbose=0):
        """
        Perform linear least squares to perturb the wavelet coefficients
        to best match their targets
        """
        M = self.targets[0].size
        for coord in range(self.dim):
            tic = time.time()
            print("Computing target coordinate {} of {}...\n".format(coord+1, self.dim))
            Y = self.coeffs[coord]
            mn = 0
            mx = np.inf
            if self.q != -1:
                eps = np.quantile(Y.flatten(), self.q)
                mn = np.maximum(0, Y.flatten()-eps)
                mx = Y.flatten()+eps
            y = np.ones(M+Y.size)
            y[0:M] = self.targets[coord]
            y[M::] = self.fit_lam*Y
            self.coeffs[coord] = lsq_linear(self.Mat, y, (mn, mx), verbose=verbose)['x']
            print("Elapsed time: {:.3f}".format(time.time()-tic))

    def get_signal(self, normalize=False):
        """
        Compute and z-normalize the sliding window sums

        Returns
        -------
        ndarray(M, dim)
            Average of the sliding window sums
        """
        M = self.targets[0].size
        X = np.zeros((M, self.dim))
        for coord in range(self.dim):
            Y = self.coeffs[coord]
            Mat = SlidingWindowSumMatrix(Y.size, self.win, self.fit_lam)
            res = Mat.dot(Y)
            x = res[0:M]
            if normalize:
                x = (x-np.mean(x))/np.std(x)
            X[:, coord] = x
        return X


##################################################
#                  WAVELET BASED
##################################################

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

class WaveletCoeffs(StegoWindowedPower):
    def __init__(self, x, target, win, fit_lam=1, wavtype='haar', wavlevel=7, coefflevel=1, q=-1):
        """
        Parameters
        ----------
        x: ndarray(N)
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
        q: float in [0, 1]
            Quantile in which to keep magnitudes.
            If -1, go up to infinity
        """
        StegoSolver.__init__(self, x, target)
        ## Step 1: Compute wavelets at all levels
        wav_coeffs = wavedec(x, wavtype, level=wavlevel)
        self.wavlevel = wavlevel
        self.coefflevel = coefflevel
        self.wav_coeffs = wav_coeffs
        self.wavtype = wavtype
        
        coeffs_mod = wavedec(wav_coeffs[coefflevel], wavtype, level=1)

        ## Step 2: Setup all aspects of sliding windows
        # Signs of wavelet coefficients before squaring
        self.signs = [np.sign(x) for x in coeffs_mod] 
        StegoWindowedPower.__init__(self, target, [p**2 for p in coeffs_mod], win, fit_lam, q)

    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting all wavelet transforms
        """
        wav_coeffs = self.wav_coeffs.copy()
        coeffs_mod = []
        for s, p in zip(self.signs, self.coeffs):
            p = np.array(p)
            p[p < 0] = 0
            coeffs_mod.append(s*np.sqrt(p))
        wav_coeffs[self.coefflevel] = waverec(coeffs_mod, self.wavtype)
        y = waverec(wav_coeffs, self.wavtype)
        return y



##################################################
#           NON-OVERLAPPING STFT BASED
##################################################
from spectrogramtools import *

class STFTPowerDisjoint(StegoWindowedPower):
    def __init__(self, x, target, win_length, freq_idxs, win, fit_lam=1, q=-1):
        """
        Parameters
        ----------
        x: ndarray(N)
            Audio samples
        target: ndarray(M, dim)
            Target curve
        win_length: int
            Window length to use in the disjoint spectrogram
        freq_idxs: list(dim)
            Which frequency indices to use for each coordinate
        win: int
            Window length to use in the sliding window power
        fit_lam: float
            Weight to put into the fit
        q: float in [0, 1]
            Quantile in which to keep magnitudes.
            If -1, go up to infinity
        """
        StegoSolver.__init__(self, x, target)
        ## Step 1: Compute wavelets at all levels
        self.win_length = win_length
        self.freq_idxs = freq_idxs
        SX = stft_disjoint(x, win_length)
        self.SXM = np.abs(SX) # Original magnitude
        self.SXP = np.arctan2(np.imag(SX), np.real(SX))

        ## Step 2: Setup all aspects of sliding windows
        StegoWindowedPower.__init__(self, target, [self.SXM[f, :] for f in freq_idxs], win, fit_lam, q)

    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting all wavelet transforms
        """
        SXM = np.array(self.SXM)
        for f, m in zip(self.freq_idxs, self.coeffs):
            SXM[f, :] = m
        return istft_disjoint(SXM*np.exp(1j*self.SXP))
