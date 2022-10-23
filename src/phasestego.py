import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from stego import *
from spectrogramtools import *
import time

class WindowedPhase(StegoSolver):
    def __init__(self, x, target, win_length, win, freq_idxs, Q, fit_lam=1):
        """
        Parameters
        ----------
        x: ndarray(N pow of 2)
            Audio samples
        target: ndarray(M, dim)
            Target curve
        win_length: int
            Window length to use in the disjoint spectrogram
        win: int
            Window length to use in sliding window sum
        freq_idxs: list of dim lists of ints
            Indices of the frequency bins that contribute to each target dimension
        Q: int
            Use 2*pi/Q range to store each coordinate
        fit_lam: float
            Weight to put into the fit
        
        """
        ## Step 1: Setup member variables
        StegoSolver.__init__(self, x, target)
        self.x_orig = np.array(x)
        self.target_orig = np.array(target)
        self.win_length = win_length
        self.win = win
        self.freq_idxs = freq_idxs
        self.fit_lam = fit_lam

        ## Step 2: Compute disjoint STFT and phase signal and put
        ## into the range [0, 1], where 0 is on a boundary between bins
        ## and 1 is halfway in between bins
        self.Q = Q
        inc = 2*np.pi/Q
        SX = stft_disjoint(x, win_length)
        self.SXM = np.abs(SX) # Magnitude
        SXP = np.arctan2(np.imag(SX), np.real(SX))
        self.SXP = SXP # Phase
        PLow = 2*(SXP - inc*np.floor(SXP/inc))/inc
        PHigh = 2*(inc*np.ceil(SXP/inc) - SXP)/inc
        X = np.minimum(PLow, PHigh)
        self.X = X
        N = X.shape[1]
        M = N-win+1

        ## Step 3: Setup all aspects of sliding windows
        self.mat = SlidingWindowSumMatrix(N, win, fit_lam) # Matrix to do sliding window averaging transforms on each phase bin
        self.targets = [] # List of normalized target time series for each coordinate (dim total)
        csm = np.array([]) # Cross-similarity matrix for aligned targets
        for coord in range(self.dim):
            targeti = np.array(target[:, coord])
            targetsi = []
            for f in freq_idxs[coord]:
                resf = self.mat.dot(self.X[f, :])[0:M]
                targetfi = np.array(target[:, coord])
                targetfi = get_normalized_target(resf, targeti, 0, np.inf)
                targetsi.append(targetfi)
                csmfi = np.abs(targetfi[:, None] - resf[None, :])
                if csm.size == 0:
                    csm = csmfi
                else:
                    csm += csmfi
            self.targets.append(targetsi)
        self.csm = csm
        self.reparam_targets_multi(csm)

    def solve(self, verbose=0):
        """
        Perform linear least squares to perturb the wavelet coefficients
        to best match their targets
        """
        M = self.targets[0][0].size
        for coord in range(self.dim):
            for i, f in enumerate(self.freq_idxs[coord]):
                tic = time.time()
                print("Computing target coordinate {} of {}, freq idx {} of {}...\n".format(coord+1, self.dim, i+1, len(self.freq_idxs[coord])))
                Y = self.X[f, :]
                y = np.ones(M+Y.size)
                y[0:M] = self.targets[coord][i]
                y[M::] = self.fit_lam*Y
                self.X[f, :] = lsq_linear(self.mat, y, (0, 1), verbose=verbose)['x']
                print("Elapsed time: {:.3f}".format(time.time()-tic))
    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting the STFT
        """
        M = self.SXM
        P = np.array(self.SXP)
        X = self.X

        inc = 2*np.pi/self.Q
        PLow = 2*(P - inc*np.floor(P/inc))/inc # Distance to bin below
        PHigh = 2*(inc*np.ceil(P/inc) - P)/inc # Distance to bin above
        LowDiff = np.abs(X-PLow)
        HighDiff = np.abs(X-PHigh)
        p = P[LowDiff < HighDiff]
        P[LowDiff < HighDiff] = inc*np.floor(p/inc) + 0.5*inc*X[LowDiff < HighDiff]
        p = P[HighDiff <= LowDiff]
        P[HighDiff <= LowDiff] = inc*np.ceil(p/inc) - 0.5*inc*X[HighDiff <= LowDiff]
        y = istft_disjoint(M*np.exp(1j*P))
        return y

    def get_target(self, normalize=False):
        M = self.targets[0][0].size
        res = np.zeros((M, self.dim))
        for coord, targetsi in enumerate(self.targets):
            targeti = targetsi[0]
            for targetfi in targetsi[1::]:
                targeti += targetfi
            if normalize:
                targeti = (targeti-np.mean(targeti))/np.std(targeti)
            res[:, coord] = targeti
        return res

    def get_signal(self, normalize=False):
        """
        Compute and z-normalize the sliding window centroids

        Returns
        -------
        ndarray(M, dim)
            Average of the sliding window centroids
        """
        M = self.targets[0][0].size
        X = np.zeros((M, self.dim))
        for coord in range(self.dim):
            res = np.zeros((len(self.freq_idxs[coord]), M))
            for i, f in enumerate(self.freq_idxs[coord]):
                resf = self.mat.dot(self.X[f, :])[0:M]
                std = np.std(resf)
                if normalize:
                    if std == 0:
                        resf = np.nan*np.ones(resf.size)
                    else:
                        resf = (resf-np.mean(resf))/std
                res[i, :] = resf
            X[:, coord] = np.nanmedian(res, axis=0)
        return X
