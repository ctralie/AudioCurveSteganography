from unittest import TestResult
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from pywt import wavedec, waverec, dwt, idwt
from stego import *
import time
from sklearn.decomposition import PCA

class StegoWindowedPower(StegoSolver):
    """
    A class for doing sliding window power on some nonnegative coefficients
    """
    def __init__(self, target, coeffs, win, fit_lam=1, q=-1, rescale_targets=True, do_viterbi=2):
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
        rescale_targets: bool
            If true, solve for a and be to best match signals_i = a*targets_i+b
        do_viterbi: bool
            Whether to do Viterbi to find a better path
        """
        self.win = win
        self.fit_lam = fit_lam
        self.q = q
        self.target_orig = np.array(target)
        self.dim = target.shape[1]
        assert(self.dim == len(coeffs))

        ## Step 2: Setup all aspects of sliding windows
        self.coeffs_orig = [np.array(c) for c in coeffs]
        coeffs = [np.array(c) for c in coeffs]
        self.coeffs = coeffs
        N = len(self.coeffs[0])
        Mat = SlidingWindowSumMatrix(N, win, fit_lam) # Matrix to do sliding window averaging transforms on each group of coefficients (dim total)
        self.Mat = Mat
        signals = []
        for coord in range(self.dim):
            Y = self.coeffs[coord]
            M = Y.size-win+1
            res = Mat.dot(Y)
            signals.append(res[0:M])
        self.targets = [target[:, k] for k in range(target.shape[1])]
        self.setup_targets(signals, rescale_targets, do_viterbi)
        
    def solve(self, verbose=0, mn=0, mx=np.inf, use_constraints=True):
        """
        Perform linear least squares to perturb the coefficients
        to best match their targets

        Parameters
        ----------
        mn: float
            Default minimum constraint
        mx: float
            Default maximum constraint
        use_constraints: bool   
            Whether to use the constraints.  If false, simply perform
            an unbounded least squares
        """
        M = self.targets[0].size
        for coord in range(self.dim):
            tic = time.time()
            print("Computing target coordinate {} of {}...\n".format(coord+1, self.dim))
            Y = self.coeffs[coord]
            if self.q != -1:
                eps = np.quantile(Y.flatten(), self.q)
                mn = np.maximum(0, Y.flatten()-eps)
                mx = Y.flatten()+eps
            y = np.ones(M+Y.size)
            y[0:M] = self.targets[coord]
            y[M::] = self.fit_lam*Y
            if use_constraints:
                self.coeffs[coord] = lsq_linear(self.Mat, y, (mn, mx), verbose=verbose)['x']
            else:
                self.coeffs[coord] = lsq_linear(self.Mat, y, verbose=verbose)['x']
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
                x = x - np.min(x)
                x = x/np.max(x)
            X[:, coord] = x
        return X



##################################################
#           NON-OVERLAPPING STFT BASED
##################################################
from spectrogramtools import *

class STFTPowerDisjoint(StegoWindowedPower):
    def __init__(self, x, target, win_length, freq_idxs, win, fit_lam=1, q=-1, rescale_targets=True, do_viterbi=True, Q=4):
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
        rescale_targets: bool
            If true, solve for a and be to best match signals_i = a*targets_i+b
        do_viterbi: bool
            Whether to do Viterbi to find a better path
        Q: int
            Use 2*pi/Q range to store each phase encoded component
        """
        ## Step 1: Compute STFT and setup magnitude 
        self.win_length = win_length
        self.freq_idxs = freq_idxs
        SX = stft_disjoint(x, win_length)
        self.SXM = np.abs(SX) # Original magnitude

        ## Step 2: Setup sliding windows for magnitude components, with target
        ## the shape of the signal
        target -= np.min(target, axis=0)[None, :]
        target /= np.max(target)
        StegoSolver.__init__(self, x, target)
        self.MagSolver = StegoWindowedPower(target, [self.SXM[f, :] for f in freq_idxs], win, fit_lam, q, rescale_targets=rescale_targets, do_viterbi=do_viterbi)

        ## Step 3: Setup solvers phase components, with target a repeated constant
        ## indicating the scale of each component
        SXP = np.arctan2(np.imag(SX), np.real(SX))
        self.SXP = SXP
        self.Q = Q
        SXP = self.SXP[self.freq_idxs, :]
        inc = 2*np.pi/Q
        PLow = 2*(SXP - inc*np.floor(SXP/inc))/inc
        PHigh = 2*(inc*np.ceil(SXP/inc) - SXP)/inc
        P = np.minimum(PLow, PHigh)
        scales = 0.5*np.max(target, axis=0) + 0.25
        scales = scales[None, :]*np.ones((P.shape[1], 1))
        self.PhaseSolver = StegoWindowedPower(scales, P, 1, fit_lam, q, rescale_targets=False, do_viterbi=False)

    def solve(self):
        self.MagSolver.solve(mn=0, mx=np.inf, use_constraints=True)
        self.PhaseSolver.solve(mn=0, mx=1, use_constraints=True)
    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting the STFT
        """
        ## Step 1: Recover magnitude components
        SXM = np.array(self.SXM)
        for f, m in zip(self.freq_idxs, self.MagSolver.coeffs):
            SXM[f, :] = m
        
        ## Step 2: Recover phase components
        SXP = np.array(self.SXP)
        P = np.array(SXP[self.freq_idxs, :])
        X = np.array(self.PhaseSolver.coeffs)
        inc = 2*np.pi/self.Q
        PLow = 2*(P - inc*np.floor(P/inc))/inc # Distance to bin below
        PHigh = 2*(inc*np.ceil(P/inc) - P)/inc # Distance to bin above
        LowDiff = np.abs(X-PLow)
        HighDiff = np.abs(X-PHigh)

        p = P[LowDiff < HighDiff]
        P[LowDiff < HighDiff] = inc*np.floor(p/inc) + 0.5*inc*X[LowDiff < HighDiff]
        p = P[HighDiff <= LowDiff]
        P[HighDiff <= LowDiff] = inc*np.ceil(p/inc) - 0.5*inc*X[HighDiff <= LowDiff]
        SXP[self.freq_idxs, :] = P

        return istft_disjoint(SXM*np.exp(1j*SXP))
    
    def get_signal(self, normalize=True):
        """
        Reconstruct the magnitude components and scale them by the 
        average phases
        """
        res = self.MagSolver.get_signal(normalize=normalize)
        if normalize:
            scale = self.PhaseSolver.get_signal(normalize=False)
            scales = np.median(scale, axis=0)
            scales -= 0.25
            scales /= np.max(scales)
            res = res*scales[None, :]
        return res

    def get_target(self, normalize=True):
        res = self.MagSolver.targets
        res = np.array(res).T
        if normalize:
            res -= np.min(res, axis=0)[None, :]
            res /= np.max(res, axis=0)[None, :]
            scales = np.median(self.PhaseSolver.target_orig, axis=0)
            scales -= 0.25
            scales /= np.max(scales)
            res = res*scales[None, :]
        return res



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
    def __init__(self, x, target, win, fit_lam=1, wavtype='haar', wavlevel=7, coefflevel=1, q=-1, do_viterbi=True):
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
        do_viterbi: boolean
            Whether or not to do viterbi coding to find a better path
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
        StegoWindowedPower.__init__(self, target, [p**2 for p in coeffs_mod], win, fit_lam, q, do_viterbi=do_viterbi)

    
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
#     NON-OVERLAPPING COMPLEX STFT PCA-BASED
##################################################

def get_rotated_distortion(Y, Z, flip, theta):
    """
    Compute the distortion between one centered curve
    and its rotated version

    Parameters
    ----------
    Y: ndarray(N, 2)
        Target
    Z: ndarray(N, 2)
        Reconstructed
    flip: boolean
        Whether to flip z
    theta: float
        Angle (in radians) by which to rotate Z

    Returns
    -------
    ZRot: ndarray(N, 2)
        Rotated/centered Z
    distortion: float
        Distortion
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    Z = Z.dot(R)
    if flip:
        Z[:, 0] *= -1
    Z = Z - np.min(Z, axis=0)[None, :]
    Z = Z/np.max(Z, axis=0)[None, :]
    Z -= np.mean(Z, axis=0)[None, :]
    d = np.sqrt(np.sum((Y-Z)**2, axis=1))
    return Z, np.mean(d)

class STFTPowerDisjointPCA(StegoWindowedPower):
    def __init__(self, x, target, sr, win_length, min_freq, max_freq, win, fit_lam=1, q=-1, pca=None, do_viterbi=True):
        """
        Parameters
        ----------
        x: ndarray(N)
            Audio samples
        target: ndarray(M, dim)
            Target curve
        sr: int
            Sample rate
        win_length: int
            Window length to use in the disjoint spectrogram
        min_freq: float
            Minimum frequency to use (in hz)
        max_freq: float
            Maximum frequency to use (in hz)
        win: int
            Window length to use in the sliding window power
        fit_lam: float
            Weight to put into the fit
        q: float in [0, 1]
            Quantile in which to keep magnitudes.
            If -1, go up to infinity
        do_viterbi: boolean
            Whether or not to do viterbi coding to find a better path
        """
        StegoSolver.__init__(self, x, target)
        ## Step 1: Compute STFT
        self.win_length = win_length
        SX = stft_disjoint(x, win_length)
        self.SX = np.array(SX)
        f1 = int(min_freq*win_length/sr)
        f2 = int(max_freq*win_length/sr)
        self.f1 = f1
        self.f2 = f2
        SX = SX[f1:f2, :]
        SXStack = np.concatenate((np.real(SX), np.imag(SX)), axis=0)
        dim = target.shape[1]
        if not pca:
            print("Making PCA")
            pca = PCA(n_components=dim)
            pca.fit(SXStack.T)
        Y = pca.transform(SXStack.T)
        self.Y_orig = np.array(Y)
        self.pca = pca

        ## Step 2: Setup all aspects of sliding windows
        StegoWindowedPower.__init__(self, target, Y.T, win, fit_lam, q, do_viterbi=do_viterbi)

    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting the STFT
        """
        f1 = self.f1
        f2 = self.f2
        M = f2-f1
        SX = np.array(self.SX)
        orig = self.pca.inverse_transform(self.Y_orig).T
        orig = orig[0:M, :] + 1j*orig[M::, :]
        SX[f1:f2, :] -= orig
        Y = np.array(self.coeffs).T
        perturbed = self.pca.inverse_transform(Y).T
        perturbed = perturbed[0:M, :] + 1j*perturbed[M::, :]
        SX[f1:f2, :] += perturbed
        return istft_disjoint(SX)
    
    def get_transformed_distortion(self):
        """
        Compute the best match of the reconstructed signal to the
        target, up to a rotation/flip.scale

        Returns
        -------
        ZRot: ndarray(N, 2)
            Rotated/centered Z
        distortion: float
            Distortion
        """
        Z = self.get_signal()
        Z -= np.mean(Z, axis=0)[None, :]
        Y = self.get_target(normalize=True)
        Y = Y - np.mean(Y, axis=0)[None, :]

        min_d = np.inf
        min_flip = False
        min_theta = 0
        for flip in [False, True]:
            for theta in np.linspace(0, 2*np.pi, 50):
                plt.clf()
                d = get_rotated_distortion(Y, Z, flip, theta)[1]
                if d < min_d:
                    min_d = d
                    min_flip = flip
                    min_theta = theta

        # Now do golden sections search
        a = min_theta - 0.2
        b = min_theta + 0.2
        gr = (np.sqrt(5)+1)/2
        c = b - (b-a)/gr
        d = a + (b-a)/gr
        fit_fn = lambda theta: get_rotated_distortion(Y, Z, min_flip, theta)[1]
        max_iters =  50
        for it in range(max_iters):
            x = fit_fn(c)
            y = fit_fn(d) 
            if x < y:
                b = d
            else:
                a = c
            c = b - (b-a)/gr
            d = a + (b-a)/gr
        min_theta = (a+b)/2
        return get_rotated_distortion(Y, Z, min_flip, min_theta)