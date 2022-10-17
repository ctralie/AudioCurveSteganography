import numpy as np

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

