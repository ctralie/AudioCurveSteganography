import matplotlib.pyplot as plt
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
        Compute and z-normalize the sliding window sums

        Returns
        -------
        ndarray(M, dim)
            Average of the sliding window sums
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








class WaveletCoeffsAvg(StegoSolver):
    """
    Similar to the WaveletCoeffs class, except the optimization is performed in
    different bands, and an average is taken at the end
    """
    def __init__(self, x, target, win, fit_lam=1, k=2, wavtype='haar', wavlevel=7, viterbi_K=-1):
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
        k: int
            Index of the wavelet to split up
        wavtype: string
            Type of wavelet to use
        wavlevel: int
            Level of wavelets to go down to
        viterbi_K: int
            K to use with Viterbi.  If -1, loop through until
            the path goes through at least one cycle
        
        """
        StegoSolver.__init__(self, x, target)
        self.x_orig = np.array(x)
        self.target_orig = np.array(target)

        ## Step 1: Determine how far to subdivide the target wavelet
        ## So that the length of each component best matches the target length
        M = target.shape[0]
        subdiv = np.ceil(np.log2( (len(x)/2**(k+1))/M ))
        div = 2**(k+1+subdiv)
        N = int(np.ceil(len(x)/div)*div)
        x2 = np.zeros(N)
        x2[0:len(x)] = x # Zeropad
        x = x2

        ## Step 2: Perform the wavelet transform
        coeffs = wavedec(x, wavtype, level=wavlevel)
        coeffs = [c for c in coeffs[::-1]]
        coeffs[k] = subdivide_wavlevel_dec(coeffs[k], subdiv+1, wavtype)
        self.wavlevel = wavlevel
        self.coeffs = coeffs
        self.k = k
        self.wavtype = wavtype
        self.win = win
        self.fit_lam = fit_lam

        ## Step 3: Setup all aspects of sliding windows
        self.signs = [np.sign(x) for x in coeffs[k]] # Signs of wavelet coefficients before squaring
        self.pairs_sqr = [p**2 for p in coeffs[k]] # Squared wavelet coefficients
        self.targets = [] # Normalized target time series for each coordinate (dim total)
        self.coords = [] # Coordinate indices of each group of coefficients
        self.Mat = SlidingWindowSumMatrix(len(self.pairs_sqr[0]), win, fit_lam) # Matrix to do sliding window averaging transforms on each group of coefficients
        
        csm = np.array([]) # Cross-similarity matrix for aligned targets
        K = len(self.signs)//self.dim
        for coord in range(self.dim):
            self.coords += [coord]*K
            self.targets.append([])
            for k in range(K):
                Y = self.pairs_sqr[coord*K+k]
                M = Y.size-win+1
                res = self.Mat.dot(Y)
                res = res[0:M]
                targeti = np.array(target[:, coord])
                targeti = get_normalized_target(res, targeti, 0, np.inf)
                self.targets[coord].append(targeti)
                csmi = np.abs(targeti[:, None] - res[None, :])
                if csm.size == 0:
                    csm = csmi
                else:
                    csm += csmi
        self.csm = csm
        self.coords = np.array(self.coords, dtype=int)
        self.reparam_targets_multi(csm, viterbi_K)
        
    def solve(self, verbose=0):
        """
        Perform linear least squares to perturb the wavelet coefficients
        to best match their targets
        """
        M = self.targets[0][0].size
        K = len(self.signs)//self.dim
        for coord in range(self.dim):
            for i in range(K):
                idx = coord*K + i
                tic = time.time()
                print("Computing target coordinate {} of {}, component {} of {}...\n".format(coord+1, self.dim, i+1, K))
                Y = self.pairs_sqr[idx]
                y = np.ones(M+Y.size)
                y[0:M] = self.targets[coord][i]
                y[M::] = self.fit_lam*Y
                self.pairs_sqr[idx] = lsq_linear(self.Mat, y, (0, np.inf), verbose=verbose)['x']
                print("Elapsed time: {:.3f}".format(time.time()-tic))
    
    def reconstruct_signal(self):
        """
        Return the 1D time series after inverting all wavelet transforms
        """
        coeffs = self.coeffs.copy()
        coeffs_mod = []
        for s, p in zip(self.signs, self.pairs_sqr):
            p = np.array(p)
            p[p < 0] = 0
            coeffs_mod.append(s*np.sqrt(p))
        coeffs[self.k] = subdivide_wavelet_rec(coeffs_mod, self.wavtype)
        y = waverec(coeffs[::-1], self.wavtype)
        return y

    def get_targets(self, normalize=False):
        """
        Return the lists of targets for each dimension

        Parameters
        ----------
        normalize: boolean
            Whether to z-normalize each component
        
        Returns
        -------
        list of list of ndarray(N)
            Target components
        """
        ret = []
        for coord, targets in enumerate(self.targets):
            ret.append([])
            for target in targets:
                if normalize:
                    target = (target - np.mean(target))
                    target = target/np.std(target)
                ret[coord].append(target)
        return ret

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

    def get_signals(self, normalize=False):
        """
        Compute and (possibly) z-normalize the sliding window sum
        lists for each dimension

        Returns
        -------
        list of list of ndarray(M)
            Each sliding window
        """
        M = self.targets[0][0].size
        K = len(self.signs)//self.dim
        ret = []
        for coord in range(self.dim):
            ret.append([])
            for i in range(K):
                idx = coord*K + i
                Y = self.pairs_sqr[idx]
                res = self.Mat.dot(Y)
                x = res[0:M]
                if normalize:
                    x = (x-np.mean(x))/np.std(x)
                ret[coord].append(x)
        return ret


    def get_signal(self, normalize=False):
        """centroid
        Compute and (possibly) z-normalize the sliding window sums

        Returns
        -------
        ndarray(M, dim)
            Average of the sliding window sums
        """
        signals = self.get_signals()
        M = signals[0][0].size
        Z = np.zeros((M, len(signals)))
        for coord, signalsi in enumerate(signals):
            signali = signalsi[0]
            for signalfi in signalsi[1::]:
                signali += signalfi
            if normalize:
                signali = (signali - np.mean(signali))/np.std(signali)
            Z[:, coord] = signali
        return Z

    def plot(self, normalize=False):
        Y = self.get_targets(normalize)
        Z = self.get_signals(normalize)
        res = 4
        K = len(self.signs)//self.dim
        rows = K*self.dim + 2
        plt.figure(figsize=(res*3, res*rows))
        for coord in range(self.dim):
            for i in range(K):
                idx = coord*K + i
                plt.subplot2grid((rows, 2), (idx, 0), colspan=2)
                plt.plot(Y[coord][i])
                plt.plot(Z[coord][i])
                plt.legend(["Target", "Signal"])
                plt.title("Coord {} Component {}".format(coord+1, i+1))
        """
        plt.subplot2grid((rows, 2), (rows-1, 0))
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
        """