import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from stego import *
from spectrogramtools import *
import time

class WindowedSpecCentroid(StegoSolver):
    def __init__(self, x, target, win_length, win, freq_idxs, denom_lam=1, fit_lam=1, q=0.5):
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
        denom_lam: float
            The weight to put on the denominator
        fit_lam: float
            Weight to put into the fit
        q: float in [0, 1]
            Quantile in which to keep magnitudes
        
        """
        ## Step 1: Setup member variables
        StegoSolver.__init__(self, x, target)
        self.x_orig = np.array(x)
        self.target_orig = np.array(target)
        self.win_length = win_length
        self.win = win
        self.freq_idxs = freq_idxs
        self.denom_lam = denom_lam
        self.fit_lam = fit_lam
        self.q = q

        ## Step 2: Compute disjoint STFT and phase signal and put
        ## into the range [0, 1], where 0 is on a boundary between bins
        ## and 1 is halfway in between bins
        SX = stft_disjoint(x, win_length)
        self.SXM = np.abs(SX) # Original magnitude
        self.X = np.array(self.SXM) # Updated magnitude
        SXP = np.arctan2(np.imag(SX), np.real(SX))
        self.SXP = SXP # Phase

        ## Step 3: Setup all aspects of sliding windows
        self.mats = [] # Matrices to do sliding window centroids on each coordinate chunk
        self.targets = [] # List of normalized target time series for each coordinate (dim total)
        csm = np.array([]) # Cross-similarity matrix for aligned targets
        M = self.SXM.shape[1]-self.win+1
        for coord in range(self.dim):
            Xi = self.SXM[self.freq_idxs[coord], :]
            mat = SlidingWindowCentroidMatrix(Xi, win, denom_lam, fit_lam)
            self.mats.append(mat)
            res = mat.dot(Xi.flatten())[M:2*M]
            targeti = np.array(target[:, coord])
            targeti = get_normalized_target(res, targeti, 0, len(self.freq_idxs[coord]))
            self.targets.append(targeti)
            csmi = np.abs(targeti[:, None] - res[None, :])
            if csm.size == 0:
                csm = csmi
            else:
                csm += csmi
        self.csm = csm
        self.reparam_targets(csm)

    def solve(self, verbose=0):
        """
        Perform linear least squares to perturb the wavelet coefficients
        to best match their targets
        """
        M = self.targets[0].size
        for coord in range(self.dim):
            print("Computing target coordinate {} of {}...\n".format(coord+1, self.dim))
            tic = time.time()
            idxs = self.freq_idxs[coord]
            Xi = self.SXM[idxs, :]
            shape = Xi.shape
            K = Xi.shape[0]
            Y = np.ones(2*M+Xi.size)
            Y[0:M] = (K/2)*self.denom_lam
            Y[M:2*M] = self.targets[coord]
            Y[2*M::] = self.fit_lam*Xi.flatten()

            eps = np.quantile(Xi.flatten(), self.q)
            mn = np.maximum(0, Xi.flatten()-eps)
            mx = Xi.flatten()+eps

            Xi = lsq_linear(self.mats[coord], Y, (mn, mx), verbose=verbose)['x']
            self.X[idxs, :] = np.reshape(Xi, shape)
            print("Elapsed time: {:.3f}".format(time.time()-tic))
    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting the STFT
        """
        y = istft_disjoint(self.X*np.exp(1j*self.SXP))
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
            idxs = self.freq_idxs[coord]
            Xi = self.X[idxs, :]
            mat = SlidingWindowCentroidMatrix(Xi, self.win, self.denom_lam, self.fit_lam)
            res = mat.dot(Xi.flatten())
            x = res[M:2*M]
            if normalize:
                x = (x-np.mean(x))/np.std(x)
            X[:, coord] = x
        return X
