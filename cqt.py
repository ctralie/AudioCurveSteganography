"""
Translated matlab code from Tomohiki Nakamura Copyright (c) 2015
Translation copyright Chris Tralie (c) 2022
"""
import numpy as np

nextpow2 = lambda x: int(np.ceil(np.log2(x)))

def lognormal_wavelet(a, sigma, T, P, alpha):
    """
    Return DFTs of scaled log-normal wavelets

    Parameters
    ----------
    a: ndarray(K)
        scales
    sigma: float
        Variance
    T: int
        Signal length
    P: float
        Bandwidth parameter for fast approximate consant-Q
        transform.  Elements of range [-P\sigma, P\sigma] are used
    alpha: float
        Log-normal wavelet parameter.  For example, if alpha=1 (2),
        time slice of magnitude (squared, resp.) wavelet transform
        of a sinusoid approximately equals a Gaussian-like function
    
    Returns
    -------
    w: ndarray(K, n_time_shifts)
        Wavelets
    leftside: ndarray(K, dtype=int)
        Left side fo computed range in the angular linear frequency domain
    sumfilout: ndarray(T//2+1)
        Normalization constant for inverse transform
    """
    K = len(a) # Num of scales
    delta_rad = 2*np.pi/T # Radians
    center_omega = 1/a # Radians
    calculate_range = np.exp([np.log(center_omega)-P*sigma, np.log(center_omega)+P*sigma])
    calculate_index = np.floor(calculate_range/delta_rad).astype(int) # Linear frequency index
    leftside = calculate_index[0, :]
    leftside[leftside < 0] = 0

    # Linear frequency index size (length of the longest wavelet)
    mx = np.max(calculate_index[1, :] - calculate_index[0, :])
    D = 2**nextpow2(mx) 
    D = int(D)
    calculate_omega = np.array(leftside[:, None] + np.arange(D)[None, :], dtype=float)
    calculate_omega *= delta_rad # Radians (K x D)
    w = np.exp(-np.log(calculate_omega*a[:, None])**2/(2*alpha*sigma**2))
    w *= (calculate_omega > 0)*(calculate_omega <= np.pi)

    # Summation of filter outputs
    sumfilout = np.zeros(T)
    for k in range(K):
        sumfilout[leftside[k]:leftside[k]+D] += np.abs(w[k, :])**2
    sumfilout = sumfilout[0:T//2+1]
    return w, leftside, sumfilout


class cqt:
    def __init__(self, T, fs, resol=24, LF=27.5, P=1.0, alpha=1.0, sigma=0.02):
        """
        Parameters
        ----------
        T: int
            Signal length
        fs: int
            Sampling frequency
        resol: int
            Number of frequency bins in each octave
        LF: float
            Lowest center frequency
        P: float
            Bandwidth parameter for fast approximate consant-Q
            transform.  Elements of range [-P\sigma, P\sigma] are used
        alpha: float
            Log-normal wavelet parameter.  For example, if alpha=1 (2),
            time slice of magnitude (squared, resp.) wavelet transform
            of a sinusoid approximately equals a Gaussian-like function
        """
        max_n = np.floor(np.log2(fs/2/LF)*resol) # number of frequency bins
        analysis_freq = LF*2**(np.arange(max_n+1)/resol); # center frequencies
        a = fs/(2*np.pi*analysis_freq) # corresponding scales
        w, leftside, sumfilout = lognormal_wavelet(a,sigma,T,P,alpha)
        D = w.shape[1]

        # Normalization for CQT
        sumfilout2 = np.array(sumfilout)
        sumfilout2[sumfilout < 1e-4] = 1e10
        norm_C = np.zeros(T)
        norm_C[0:sumfilout2.size] = 1/sumfilout2
        if T%2 == 0:
            norm_C[T//2::] = 0
        else:
            norm_C[int(np.ceil(T/2))-1::] = 0

        # Save away local variables
        self.resol = resol
        self.sigma = sigma
        self.a = a
        self.w = w
        self.L = leftside
        self.calculate_indexes = self.L[:, None] + np.array(np.arange(D)[None, :], dtype=int)
        self.T = T
        self.fs = fs
        self.timeshift = T/fs/D
        self.index2time = self.timeshift*np.arange(D)
        self.index2freqHz = analysis_freq
        self.sumfilout = sumfilout
        self.norm_C = norm_C

    def forward(self, x):
        """
        Perform a forward fast wavelet transform on a signal of compatible length

        Parameters
        ----------
        x: ndarray(T)
            Signal
        """
        T = self.T; assert(x.size == T)
        D = self.w.shape[1]
        FTx1 = np.fft.fft(x)
        FTx1[T//2::] = 0
        spec = FTx1[self.calculate_indexes]*self.w
        spec = np.fft.ifft(spec, axis=1)
        spec *= np.exp(1j*2*np.pi*self.L[:, None]*np.arange(D)[None, :]/D)
        return spec
    
    def inverse(self, S):
        K = S.shape[0]
        assert(K == self.w.shape[0])
        D = self.w.shape[1]
        assert(D == self.w.shape[1])
        L = self.L
        S = S*np.exp(-1j*2*np.pi*self.L[:, None]*np.arange(D)[None, :]/D)
        S = np.fft.fft(S, axis=1)*np.conj(self.w)
        # Now do shift overlap/add
        spec = np.zeros(self.T, dtype=S.dtype)
        for k in range(K):
            spec[L[k]:L[k]+D] += S[k, :]
        spec = spec*self.norm_C
        n = spec.size//2
        spec[n+1::] = np.conj(spec[1:n][::-1])
        return np.fft.ifft(spec)
    
    def griffin_lim(self, S, n_iters=20):
        """
        Perform Griffin-Lim phase retrieval on a magnitude
        CQT-spectrogram S
        """
        A = np.array(S, dtype=complex)
        A = A*np.exp(1j*2*np.pi*np.random.rand(A.shape[0], A.shape[1]))
        for i in range(n_iters):
            print(".", end='')
            P = self.forward(np.real(self.inverse(A)))
            angle = np.arctan2(np.imag(P), np.real(P))
            A = S*np.exp(1j*angle)
        return np.real(self.inverse(A))
