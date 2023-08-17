"""
Programmer: Chris Tralie
Purpose: A wrapper around the pywavelet toolbox (pywt)
to perform some wavelet operations on images.  Images are assumed
to have dimensions that are a power of 2
"""

import pywt
import numpy as np

def wav_forward(I, n_levels):
    """
    Perform a forward wavelet transform down to some level

    Parameters
    ----------
    I: ndarray(M, N)
        Image
    n_levels: int
        Number of levels of wavelet transform
    
    Returns
    -------
    list of coefficients
    """
    coeffs = []
    LL = I
    for i in range(n_levels):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'db1')
        coeffs.append((LH, HL, HH))
    coeffs.append(LL)
    return coeffs
        
def wav_backward(pcoeffs):
    """
    Perform an inverse wavelet transform

    Parameters
    ----------
    pcoeffs: list of ndarray(,)
        List of wavelet coefficients
    
    Returns
    -------
    ndarray(M, N)
        Image
    """
    coeffs = pcoeffs.copy()
    while len(coeffs) > 1:
        LL, (LH, HL, HH) = coeffs.pop(), coeffs.pop()
        coeffs.append(pywt.idwt2((LL, (LH, HL, HH)), 'db1'))
    return coeffs[0]


def wav_arrange_img(pcoeffs):
    """
    Arrange the wavelet coefficients into an image

    Parameters
    ----------
    pcoeffs: list of ndarray(,)
        List of wavelet coefficients
    
    Returns
    -------
    ndarray(M, N)
        Wavelet coefficients
    """
    coeffs = pcoeffs.copy()
    while len(coeffs) > 1:
        LL, (LH, HL, HH) = coeffs.pop(), coeffs.pop()
        M, N = LL.shape[0], LL.shape[1]
        J = np.zeros((M*2, N*2))
        J[0:M, 0:N] = LL
        J[0:M, N::] = LH
        J[M::, 0:N] = HL
        J[M::, N::] = HH
        coeffs.append(J)
    return J

def wav_extract_img(I, n_levels):
    """
    Take the wavelet coefficients out of an image and put
    them back into a list for pywt

    Parameters
    ----------
    I: ndarray(M, N)
        Wavelet coefficients
    
    Returns
    -------
    Wavelet coefficients
    """
    coeffs = []
    J = I
    for i in range(n_levels):
        M = J.shape[0]//2
        N = J.shape[1]//2
        LL = J[0:M, 0:N]
        LH = J[0:M, N::]
        HL = J[M::, 0:N]
        HH = J[M::, N::]
        coeffs.append((LH, HL, HH))
        J = LL
    coeffs.append(J)
    return coeffs

def get_color_wavelet_tsp(I, n_levels, n_points, box_scale=1):
    """
    Parameters
    ----------
    I: ndarray(N, N, 3)
        Square color image; assumed to have dimensions a power of 2
    n_levels: int
        Number of levels in the wavelet transform
    n_points: int
        Number of points to have in the tsp tour
    box_scale: float
        Put the x and y coordinates into the square [0, box_scale] x [0, box_scale]
    
    Returns
    -------
    xyrgb: ndarray(n_points, 5)
        The wavelet coefficient TSP tour
    """
    from tsp import get_tsp_tour

    ## Step 1: Compute wavelet transform
    res = I.shape[0]//2**n_levels

    coeffs = np.array([wav_arrange_img(wav_forward(I[:, :, k], n_levels)) for k in range(3)])
    coeffs = np.moveaxis(coeffs, 0, 2)
    weights = coeffs[:, :, 0]**2 + coeffs[:, :, 1]**2 + coeffs[:, :, 2]**2

    ## Step 2: Extract the n_points largest coefficients by weight
    q = -np.partition(-weights.flatten(), n_points)[n_points]
    for k in range(3):
        coeffs[:, :, k][weights < q] = 0

    pix = np.arange(I.shape[0])
    X, Y = np.meshgrid(pix, pix)
    X = X[weights > q]
    Y = Y[weights > q]
    rgb = coeffs[weights > q, :]/2**n_levels

    xyrgb = np.array(np.concatenate((X[:, None], Y[:, None], rgb), axis=1), dtype=float)
    xyrgb[:, 0:2] *= box_scale/I.shape[0]

    def get_tsp_sub_tour(X):
        ret = X
        if X.shape[0] > 3:
            try:
                ret = get_tsp_tour(X)
            except:
                ret = X
        return ret

    ## Step 3: Do TSP in chunks
    res = I.shape[0]//2**n_levels
    tour = get_tsp_sub_tour(xyrgb[(X < res)*(Y < res), :])
    while res < I.shape[0]:
        YLH = get_tsp_sub_tour(xyrgb[(X >= res)*(X < 2*res)*(Y < res), :])
        YHL = get_tsp_sub_tour(xyrgb[(X < res)*(Y >= res)*(Y < 2*res), :])
        YHH = get_tsp_sub_tour(xyrgb[(X >= res)*(X < 2*res)*(Y >= res)*(Y < 2*res), :])
        tour = np.concatenate((tour, YLH, YHH, YHL), axis=0)
        res *= 2
    
    return tour

def invert_sparse_coefficients(xyrgb, res, n_levels, box_scale=1):
    """
    Parameters
    ----------
    xyrgb: ndarray(n_points, 5)
        The wavelet coefficient TSP tour
    res: int
        Resolution of image
    n_levels: int
        Number of levels in the wavelet transform
    box_normalize: bool
        If True (default), normalize x and y coordinates to [0, 1]
    """
    coeffs_out = np.zeros((res, res, 3))
    for i in range(xyrgb.shape[0]):
        x = int(xyrgb[i, 0]*res/box_scale)
        y = int(xyrgb[i, 1]*res/box_scale)
        coeffs_out[y, x] = xyrgb[i, 2::]*2**n_levels
    Js = [wav_backward(wav_extract_img(coeffs_out[:, :, k], n_levels)) for k in range(3)]
    J = np.zeros((res, res, 3))
    for k in range(3):
        J[:, :, k] = Js[k]
    J[J < 0] = 0
    J[J > 1] = 1
    return J