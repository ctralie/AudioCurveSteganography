from curses import A_COLOR
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import time
from scipy import sparse
from scipy.sparse import linalg as slinalg
from scipy.signal import medfilt


def get_window_energy(x, win, hop=1):
    """
    Return the sliding window squared energy of a signal

    Parameters
    ----------
    x: ndarray(N)
        Input signal
    win: int
        Window size
    hop: int
        Hop length between windows
    
    Returns
    -------
    ndarray(N-win+1)
        Windowed energy
    """
    eng = np.cumsum(np.concatenate(([0], x**2)))
    return eng[win::hop]-eng[0:-win:hop]

def energy_perturb(x, target, win, hop, lam):
    """
    Perturb the windowed energy of a signal to match a target.
    Do an unconstrained least squares, trading off fit to original
    signal and fit to target signal

    Parameters
    ----------
    x: ndarray(N)
        An array of samples
    target: ndarray(T <= N-win+1)
        Target signal
    win: int
        Length of sliding window
    hop: int
        Hop length between windows
    lam: float
        Weight of audio fidelity
    
    Returns
    -------
    dict(
        target: ndarray(T)
            The target signal, normalized to the range of cent
    )
    """
    N = x.size
    T = target.size
    assert(T <= 1+(N-win)//hop)
    NT = (T-1)*hop+win # Number of samples involved in the optimization (NT <= N)
    

    ## Step 1: Compute the windowed energy "wineng" and normalize the target signal 
    ## into the range (mu(wineng)-std(wineng), mu(wineng)+std(wineng))
    wineng =  get_window_energy(x, win, hop)[0:T]
    target -= np.min(target)
    target /= np.max(target)
    target = np.mean(wineng) + (target-0.5)*2*np.std(wineng)

    ## Step 2: Come up with system of linear equations
    I1 = np.arange(NT)
    I2 = NT + (np.arange(T)[:, None]*np.ones((1, win))).flatten()
    I = np.concatenate((I1.flatten(), I2.flatten()))

    J1 = np.arange(NT)
    J2 = hop*np.arange(T)[:, None] + np.arange(win)[None, :]
    J = np.concatenate((J1, J2.flatten()))
    
    V = np.ones(NT+T*win, dtype=int)
    V[0:NT] = lam

    A = sparse.coo_matrix((V, (I, J)), shape=(NT+T, NT))
    A = A.tocsr()
    b = np.concatenate((lam*x[0:NT]**2, target))
    
    ## Step 3: Solve system of equations
    #"""
    u = cp.Variable(NT)
    u.value = x[0:NT]
    objective = cp.Minimize(cp.sum_squares(A@u - b))
    constraints = [u >= 0]
    tic = time.time()
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True)
    print("Elapsed time least squares: ", time.time()-tic)
    u = u.value
    ubefore = np.array(u)
    u[u < 0] = 0
    #"""

    #from scipy.optimize import nnls
    #tic = time.time()
    #u = nnls(A.toarray(), b)[0]
    #ubefore = np.array(u)
    #print("Elapsed time nnls: ", time.time()-tic)

    #from sgd import stochastic_gradient_descent
    #u = stochastic_gradient_descent(b, A, x[0:T+win]**2, step_size=0.05, converge_on_r=0.1, non_neg=True)
    #ubefore = np.array(u)
    
    xres = np.array(x)
    xres[0:NT] = np.sign(x)[0:NT]*np.sqrt(u)
    
    return dict(target=target, wineng=wineng, x=xres, u=ubefore)


def get_windowed_spec_centroid_matrices(Mag, min_freq, max_freq, win):
    """
    Get the sparse matrices to compute the numerator and the denominator
    of the windowed spectral centroid

    Parameters
    ----------
    Mag: ndarray(n_freq, N)
        A magnitude spectrogram
    min_freq: int
        Minimum frequency index to use
    max_freq: int
        One beyond the maximum frequency index to use
    win: int
        Length of sliding window
    
    Returns
    -------
    An: sparse array(N-win+1, N*M)
        Numerator matrix
    Ad: sparse array(N-win+1, N*M)
        Denominator matrix
    Ac: sparse array(N, N*M)
        Individual denominator matrix
    """
    In = []
    Jn = []
    Vn = []

    Id = []
    Jd = []
    Vd = []

    N = Mag.shape[1]
    M = max_freq-min_freq
    for i in range(N-win+1):
        # Setup spectral numerator matrix
        In += [i]*M*win
        Jn += np.arange(i*M, (i+win)*M).tolist()
        Vi = np.ones((win, 1))*np.arange(min_freq, max_freq)[None, :]
        Vi = Vi/np.sum(Mag[min_freq:max_freq, i:i+win])
        Vn += (Vi.flatten()).tolist()
        # Setup spectral centroid denominator matrix
        Id += [i]*M*win
        Jd += np.arange(i*M, (i+win)*M).tolist()
        Vd += np.ones(M*win).tolist()


    An = sparse.coo_matrix((Vn, (In, Jn)), shape=(N-win+1, N*M))
    Ad = sparse.coo_matrix((Vd, (Id, Jd)), shape=(N-win+1, N*M))
    return An, Ad

def windowed_spec_centroid_perturb(Mag, target, win, min_freq, max_freq, eps_add, eps_ratio=0):
    """
    Perturb a sliding window of the spectral centroid of a magnitude spectrogram within 
    a particular frequency range to best match some target signal, up to a scale

    Parameters
    ----------
    Mag: ndarray(n_freq, N)
        A magnitude spectrogram
    target: ndarray(T >= N)
        Target signal
    win: int
        Length of sliding window
    min_freq: int
        Minimum frequency index to use
    max_freq: int
        One beyond the maximum frequency index to use
    eps_add: float
        Maximum amount of additive perturbation allowed at each frequency bin
    eps_ratio: float
        Maximum amount of ratio perturbation allowed at each frequency bin
    
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

    ## Step 1: Setup sparse optimization problem for perturbing the magnitude 
    ## spectrogram values so that the distance between the target and the 
    ## windowed spectral centroid is minimized in a least squared sense
    M = max_freq - min_freq
    N = Mag.shape[1]
    An, Ad = get_windowed_spec_centroid_matrices(Mag, min_freq, max_freq, win)
    
    ## Step 2: Normalize the target so it's within a reasonable range of
    ## the current windowed spectral centroid
    x_orig = (Mag[min_freq:max_freq, :].T).flatten()
    cent_orig = An.dot(x_orig)
    xrg = 2*np.std(cent_orig)
    xmu = np.mean(cent_orig)
    xmin = max(min_freq, xmu-xrg)
    xmax = min(max_freq, xmu+xrg)

    target -= np.min(target)
    target /= np.max(target)
    target = xmin + target*(xmax-xmin)
    b = target[0:N-win+1]

    ## Step 3: Solve the optimization problem and return a spectrogram
    ## with the result
    #from opt import lsq_box
    #tic = time.time()
    #x, resid = lsq_box(A, b, )

    #"""
    x = cp.Variable(N*M)
    objective = cp.Minimize(cp.sum_squares(An@x - b))
    constraints = [Ad@x == Ad.dot(x_orig)]
    xmin = x_orig - eps_add
    xmax = x_orig + eps_add
    if eps_ratio > 0:
        xmin = np.minimum(xmin, x_orig*(1-eps_ratio))
        xmax = np.maximum(xmax, x_orig*(1+eps_ratio))
    xmin[xmin < 0] = 0
    constraints += [xmin <= x, x <= xmax]
    tic = time.time()
    prob = cp.Problem(objective, constraints)
    prob.solve()
    #"""


    print("Elapsed Time: ", time.time()-tic)
    x = np.array(x.value)
    MagRet = np.array(Mag)
    MagRet[min_freq:max_freq, :] = np.reshape(x, (N, M)).T


    x = (MagRet[min_freq:max_freq, :].T).flatten()
    cent = An.dot(x)
    
    return dict(cent_orig=cent_orig, b=b, Mag=MagRet, cent=cent)


def spec_centroid(Mag, f1, f2, win):
    """
    Compute the windowed spectral centroid of a magnitude short time 
    frequency transform within a particular frequency range

    Parameters
    ----------
    Mag: ndarray(n_freq, N)
        A magnitude spectrogram
    f1: int
        Minimum frequency index to use
    f2: int
        One beyond the maximum frequency index to use
    win: int
        Window to use
    """
    An, _ = get_windowed_spec_centroid_matrices(Mag, f1, f2, win)
    return An.dot((Mag[f1:f2, :].T).flatten())


def plot_stego_centroid_fit(f1, f2, win, orig, fit, target, med=31):
    """
    Plot the results of perturbing the spectral centroids

    Parameters
    ----------
    f1: int
        Minimum frequency index to use
    f2: int
        One beyond the maximum frequency index to use
    win: int
        Window length used
    orig: ndarray(n_freq, N)
        Original magnitude spectrogram
    fit: ndarray(n_freq, N)
        Computed magnitude spectrogram
    target: ndarray(T >= N)
        Target signal
    med: int
        Amount by which to median filter the fitted centroid
    """
    plt.subplot(131)
    plt.imshow(orig[f1:f2, :] - fit[f1:f2, :], aspect='auto', cmap='gray')
    plt.gca().invert_yaxis()
    plt.colorbar()

    plt.subplot(132)
    h = spec_centroid(fit, f1, f2, win)
    h = medfilt(h, med)
    plt.scatter(h, target[0:h.size])
    plt.axis("equal")
                    
    plt.subplot(133)
    plt.plot(target)
    plt.plot(h)