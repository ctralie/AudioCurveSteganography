import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import time
from scipy import sparse

def spec_centroid_perturb(Mag, target, min_freq, max_freq, eps):
    """
    Perturb the spectral centroid of a magnitude spectrogram within 
    a particular frequency range to best match some target signal, up to a scale

    Parameters
    ----------
    Mag: ndarray(n_freq, N)
        A magnitude spectrogram
    target: ndarray(T >= N)
        Target signal
    min_freq: int
        Minimum frequency index to use
    max_freq: int
        One beyond the maximum frequency index to use
    eps: float
        Maximum amount of perturbation allowed at each frequency bin
    
    Returns
    -------
    dict(
        Mag: ndarray(n_freq, N)
            The modified magnitude spectrogram,
        cent: ndarray(N)
            The computed original spectral centroids in the range [min_freq, max_freq)
        target: ndarray(T)
            The target signal, normalized to the range of cent
    )
    """
    ## Step 1: Compute the spectral centroid "cent" and normalize the target signal 
    ## into the range (mu(cent)-std(cent), mu(cent)+std(cent))
    cent = np.sum(Mag[min_freq:max_freq, :]*np.arange(min_freq, max_freq)[:, None], axis=0) / np.sum(Mag[min_freq:max_freq], axis=0)
    
    target -= np.min(target)
    target /= np.max(target)
    target = np.mean(cent) + (target-0.5)*2*np.std(cent)

    ## Step 2: Setup sparse optimization problem for perturbing the magnitude spectrogram values
    ## so that the distance between the target and the spectral centroid is minimized in a least squared sense
    I = []
    J = []
    V = []
    b = []

    Ic = []
    Jc = []
    Vc = []
    norms = []

    N = Mag.shape[1]
    M = max_freq-min_freq

    x = cp.Variable(N*M)

    for i in range(N):
        # Setup spectral centroid matrix
        norm = np.sum(Mag[min_freq:max_freq, i])
        I += [i]*M
        J += np.arange(i*M, (i+1)*M).tolist()
        V += (np.arange(min_freq, max_freq)/norm).tolist()
        # Add constraints that each norm must sum to what it is now
        Ic += [i]*M
        Jc += np.arange(i*M, (i+1)*M).tolist()
        Vc += np.ones(M).tolist()
        norms.append(norm)
        b.append(target[i])


    A = sparse.coo_matrix((V, (I, J)), shape=(N, N*M))
    b = np.array(b)
    Ac = sparse.coo_matrix((Vc, (Ic, Jc)), shape=(N, N*M))
    norms = np.array(norms)
    
    ## Step 3: Solve the optimization problem and return a spectrogram
    ## with the result
    objective = cp.Minimize(cp.sum_squares(A@x - b))
    B = Mag[min_freq:max_freq, :]
    B = (B.T).flatten()
    constraints = [B - eps <= x, x <= B + eps, Ac@x == norms]
    tic = time.time()
    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("Elapsed Time: ", time.time()-tic)
    MagRet = np.array(Mag)
    MagRet[min_freq:max_freq, :] = np.reshape(np.array(x.value), (N, M)).T

    return dict(Mag=MagRet, cent=cent, target=target)