import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from numba import jit

hann = lambda win: 0.5*(1-np.cos(2*np.pi*np.arange(win)/win))

def sdct(x, win, hop, winfn=hann):
    """
    Short time discrete cosine transform

    Parameters
    ----------
    x: ndarray(N)
        Input signal
    win: int
        Window length
    hop: int
        Hop length
    winfn: function int->ndarray
        Window function
    
    Returns
    -------
    ndarray(win, 1+(N-win)//hop)
        Short time DCT spectrogram
    """
    from scipy.fft import dct
    M = 0
    if len(x) >= win:
        M = (len(x)-win)//hop + 1
    h = winfn(win)
    S = np.zeros((win, M))
    for i in range(M):
        xi = x[i*hop:i*hop+win]
        S[:, i] = dct(h*xi, type=2)
    return S

def isdct(S, hop, winfn=hann):
    """
    Inverse short time discrete cosine transform

    Parameters
    ----------
    S: ndarray(win, M)
        Short time DCT spectrogram
    hop: int
        Hop length
    winfn: function int->ndarray
        Window function
    
    Returns
    -------
    x: ndarray(win + (M-1)*hop)
        Inverse signal
    """
    from scipy.fft import idct
    win = S.shape[0]
    h = winfn(win)/(0.5*win/hop)
    N = win + hop*(S.shape[1]-1)
    X = np.zeros(N)
    for i in range(S.shape[1]):
        X[i*hop:i*hop+win] += h*idct(S[:, i])
    return X

@jit(nopython=True)
def get_maxes(S, max_freq, time_win, freq_win):
    ret = []
    M, N = S.shape
    for i in range(max_freq):
        for j in range(N):
            constraint = True
            ni = max(0, i-freq_win)
            while constraint and ni < min(max_freq, i+freq_win+1):
                nj = max(0, j-time_win)
                while constraint and nj < min(N, j+time_win+1):
                    if ni != i or nj != j:
                        if S[ni, nj] > S[i, j]:
                            constraint = False
                    nj += 1
                ni += 1
            if constraint:
                ret.append([i, j])
    return ret



halfsine = lambda W: np.sin(np.pi*np.arange(W)/float(W))
hann = lambda W: 0.5*(1 - np.cos(2*np.pi*np.arange(W)/W))
def blackman(W):
    alpha = 0.16
    a0 = (1-alpha)/2
    a1 = 0.5
    a2 = alpha/2
    t = np.arange(W)/W
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t)

def stft(X, W, H, winfunc=blackman):
    """
    :param X: An Nx1 audio signal
    :param W: A window size
    :param H: A hopSize
    :param winfunc: Handle to a window function
    """
    Q = W/H
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    if not winfunc:
        #Use half sine by default
        winfunc = halfsine
    win = winfunc(W)
    NWin = int(np.floor((X.size - W)/float(H)) + 1)
    S = np.zeros((W, NWin), dtype = np.complex)
    for i in range(NWin):
        S[:, i] = np.fft.fft(win*X[np.arange(W) + (i-1)*H])
    #Second half of the spectrum is redundant for real signals
    if W%2 == 0:
        #Even Case
        S = S[0:int(W/2)+1, :]
    else:
        #Odd Case
        S = S[0:int((W-1)/2)+1, :]
    return S

def istft(pS, W, H, winfunc=blackman):
    """
    :param pS: An NBins x NWindows spectrogram
    :param W: A window size
    :param H: A hopSize
    :param winfunc: Handle to a window function
    :returns S: Spectrogram
    """
    #First put back the entire redundant STFT
    S = np.array(pS, dtype = np.complex)
    if W%2 == 0:
        #Even Case
        S = np.concatenate((S, np.flipud(np.conj(S[1:-1, :]))), 0)
    else:
        #Odd Case
        S = np.concatenate((S, np.flipud(np.conj(S[1::, :]))), 0)
    
    #Figure out how long the reconstructed signal actually is
    N = W + H*(S.shape[1] - 1)
    X = np.zeros(N, dtype = np.complex)
    
    #Setup the window
    Q = W/H
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    if not winfunc:
        #Use half sine by default
        winfunc = halfsine
    win = winfunc(W)
    win = win/(Q/2.0)

    #Do overlap/add synthesis
    for i in range(S.shape[1]):
        X[i*H:i*H+W] += win*np.fft.ifft(S[:, i])
    return X


def stft_disjoint(x, win_length):
    """
    Make the hop length and the win length the same
    """
    M = x.size//win_length
    S = np.zeros((win_length//2+1, M), dtype=complex)
    for i in range(M):
        xi = x[i*win_length:(i+1)*win_length]
        S[:, i] = np.fft.fft(xi)[0:S.shape[0]]
    return S

def istft_disjoint(S):
    w = (S.shape[0]-1)*2
    x = np.zeros(S.shape[1]*w)
    for i in range(S.shape[1]):
        si = S[:, i]
        si = np.concatenate((si, np.conj(si[-2:0:-1])))
        xi = np.fft.ifft(si)
        x[i*w:(i+1)*w] = np.real(xi)
    return x

def change_phases(PhasesOrig, I, X):
    """
    PhasesOrig: ndarray(M, N)
        Original phases
    I: ndarray(M, N)
        Grayscale image which guides what phases should be changed to
    X: ndarray(K, 2)
        Indices of phases in PhasesOrig to change
    """
    Phases = np.array(PhasesOrig)
    for [i, j] in X:
        phase = Phases[i, j]
        # Figure out which of two options is a smaller change
        g1 = I[i, j]*np.pi # Target grayscale value
        g2 = -g1
        x = np.array([np.cos(phase), np.sin(phase)])
        x1 = np.array([np.cos(g1), np.sin(g1)])
        x2 = np.array(x1)
        x2[1] *= -1
        a1 = np.arccos(max(min(np.sum(x*x1), 1), -1))
        a2 = np.arccos(max(min(np.sum(x*x2), 1), -1))
        if a1 < a2:
            Phases[i, j] = g1
        else:
            Phases[i, j] = g2
    return Phases


def make_voronoi_image(coords, phases):
    phases = np.abs(phases)
    i1, j1 = np.min(coords, axis=0)
    i2, j2 = np.max(coords, axis=0)
    I, J = np.meshgrid(np.arange(i1, i2+1), np.arange(j1, j2+1), indexing='ij')
    shape = I.shape
    I = I.flatten()
    J = J.flatten()
    tree = KDTree(coords)
    _, idx = tree.query(np.array([I, J]).T)
    return np.reshape(phases[idx], shape)


def getNSGT(X, Fs, resol=24):
    """
    Perform a Nonstationary Gabor Transform implementation of CQT
    :param X: A 1D array of audio samples
    :param Fs: Sample rate
    :param resol: Number of CQT bins per octave
    """
    from nsgt import NSGT,OctScale
    scl = OctScale(50, Fs, resol)
    nsgt = NSGT(scl, Fs, len(X), matrixform=True)
    C = nsgt.forward(X)
    return np.array(C)

def getiNSGT(C, L, Fs, resol=24):
    """
    Perform an inverse Nonstationary Gabor Transform
    :param C: An NBinsxNFrames CQT array
    :param L: Number of samples in audio file
    :param Fs: Sample rate
    :param resol: Number of CQT bins per octave
    """
    from nsgt import NSGT,OctScale
    scl = OctScale(50, Fs, resol)
    nsgt = NSGT(scl, Fs, L, matrixform=True)
    return nsgt.backward(C)

def getiNSGTGriffinLim(C, L, Fs, resol=24, randPhase = False, NIters = 20):
    from nsgt import NSGT,OctScale
    scl = OctScale(50, Fs, resol)
    nsgt = NSGT(scl, Fs, L, matrixform=True)
    eps = 2.2204e-16
    if randPhase:
        C = np.exp(np.complex(0, 1)*np.random.rand(C.shape[0], C.shape[1]))*C
    A = np.array(C, dtype = np.complex)
    for i in range(NIters):
        print("iNSGT Griffin Lim Iteration %i of %i"%(i+1, NIters))
        Ai = np.array(nsgt.forward(nsgt.backward(C)))
        A = np.zeros_like(C)
        A[:, 0:Ai.shape[1]] = Ai
        Norm = np.sqrt(A*np.conj(A))
        Norm[Norm < eps] = 1
        A = np.abs(C)*(A/Norm)
    X = nsgt.backward(A)
    return np.real(X)