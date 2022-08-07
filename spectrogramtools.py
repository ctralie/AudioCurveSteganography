import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from numba import jit


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
