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

class WaveletCoeffs:
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
        self.x_orig = np.array(x)
        self.target_orig = np.array(target)
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

        ## Step 3: Re-parameterize targets
        viterbi_K = 1
        finished = False
        path = []
        while not finished and viterbi_K < 10:
            pathk = viterbi_loop_trace(csm, viterbi_K)
            cost1 = np.sum(csm[pathk, np.arange(csm.shape[1])])
            path2 = viterbi_loop_trace(csm[:, ::-1], viterbi_K)
            path2.reverse()
            cost2 = np.sum(csm[path2, np.arange(csm.shape[1])])
            if cost2 < cost1:
                pathk = path2
            path_unwrap = np.unwrap(pathk, period=target.shape[0])
            if np.abs(path_unwrap[0]-path_unwrap[-1]) >= target.shape[0]:
                finished = True
                path = pathk
            else:
                viterbi_K += 1
        print("viterbi_K = ", viterbi_K)
        plt.figure()
        plt.plot(path)
        self.targets = [t[path] for t in self.targets]
        
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
            Mat = SlidingWindowSumMatrix(Y.size, self.win, self.fit_lam)
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