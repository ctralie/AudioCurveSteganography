import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import lsq_linear
from pywt import wavedec, waverec, dwt, idwt


class SlidingWindowMatrix(LinearOperator):
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
