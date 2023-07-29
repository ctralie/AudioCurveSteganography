import librosa
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import subprocess
import os

################################################
# Loudness code modified from original Google Magenta DDSP implementation in tensorflow
# https://github.com/magenta/ddsp/blob/86c7a35f4f2ecf2e9bb45ee7094732b1afcebecd/ddsp/spectral_ops.py#L253
# which, like this repository, is licensed under Apache2 by Google Magenta Group, 2020
# Modifications by Chris Tralie, 2023

def power_to_db(power, ref_db=0.0, range_db=80.0, use_tf=True):
    """Converts power from linear scale to decibels."""
    # Convert to decibels.
    db = 10.0*np.log10(np.maximum(power, 10**(-range_db/10)))
    # Set dynamic range.
    db -= ref_db
    db = np.maximum(db, -range_db)
    return db

def extract_loudness(x, sr, hop_length, n_fft=512):
    """
    Extract the loudness in dB by using an A-weighting of the power spectrum
    (section B.1 of the paper)

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate (used to figure out frequencies for A-weighting)
    hop_length: int
        Hop length between loudness estimates
    n_fft: int
        Number of samples to use in each window
    """
    # Computed centered STFT
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True)
    
    # Compute power spectrogram
    amplitude = np.abs(S)
    power = amplitude**2

    # Perceptual weighting.
    freqs = np.arange(S.shape[0])*sr/n_fft
    a_weighting = librosa.A_weighting(freqs)[:, None]

    # Perform weighting in linear scale, a_weighting given in decibels.
    weighting = 10**(a_weighting/10)
    power = power * weighting

    # Average over frequencies (weighted power per a bin).
    avg_power = np.mean(power, axis=0)
    loudness = power_to_db(avg_power)
    return np.array(loudness, dtype=np.float32)

################################################


def get_filtered_noise(H, A, win_length):
    """
    Perform subtractive synthesis by applying FIR filters to windows
    and summing overlap-added versions of them together
    
    Parameters
    ----------
    H: torch.tensor(n_batches x time x n_coeffs)
        FIR filters for each window for each batch
    A: torch.tensor(n_batches x time x 1)
        Amplitudes for each window for each batch
    win_length: int
        Window length of each chunk to which to apply FIR filter.
        Hop length is assumed to be half of this
        
    Returns
    -------
    torch.tensor(n_batches, hop_length*(time-1)+win_length)
        Filtered noise for each batch
    """
    n_batches = H.shape[0]
    T = H.shape[1]
    n_coeffs = H.shape[2]
    hop_length = win_length//2
    n_samples = hop_length*(T-1)+win_length

    ## Pad impulse responses and generate noise
    H = nn.functional.pad(H, (0, win_length*2-n_coeffs))
    noise = torch.randn(n_batches, n_samples).to(H)

    ## Take out each overlapping window of noise
    N = torch.zeros(n_batches, T, win_length*2).to(H)
    n_even = n_samples//win_length
    N[:, 0::2, 0:win_length] = noise[:, 0:n_even*win_length].view(n_batches, n_even, win_length)
    n_odd = T - n_even
    N[:, 1::2, 0:win_length] = noise[:, hop_length:hop_length+n_odd*win_length].view(n_batches, n_odd, win_length)
    
    # Apply amplitude to each window
    N = N*A
    
    ## Perform a zero-phase version of each filter and window
    FH = torch.fft.rfft(H)
    FH = torch.real(FH)**2 + torch.imag(FH)**2 # Make it zero-phase
    FN = torch.fft.rfft(N)
    y = torch.fft.irfft(FH*FN)[..., 0:win_length]
    y = y*torch.hann_window(win_length).to(y)

    ## Overlap-add everything
    ola = torch.zeros(n_batches, n_samples).to(y)
    ola[:, 0:n_even*win_length] += y[:, 0::2, :].reshape(n_batches, n_even*win_length)
    ola[:, hop_length:hop_length+n_odd*win_length] += y[:, 1::2, :].reshape(n_batches, n_odd*win_length)
    
    return ola

def get_mp3_noise(X, sr):
    """
    Compute the mp3 noise of a batch of audio samples using ffmpeg
    as a subprocess
    
    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples, 1)
        Audio samples
    sr: int
        Audio sample rate
    
    Returns
    -------
    torch.tensor(n_batches, n_samples, 1)
        mp3 noise
    """
    orig_T = X.shape[1]
    X = nn.functional.pad(X, (0, X.shape[1]//4, 0, 0))
    x = X.detach().cpu().numpy().flatten()
    x = np.array(x*32768, dtype=np.int16)
    fileprefix = "temp{}".format(np.random.randint(1000000))
    wavfilename = "{}.wav".format(fileprefix)
    mp3filename = "{}.mp3".format(fileprefix)
    wavfile.write(wavfilename, sr, x)
    if os.path.exists(mp3filename):
        os.remove(mp3filename)
    subprocess.call("ffmpeg -i {} {}".format(wavfilename, mp3filename).split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    x, _ = librosa.load(mp3filename, sr=sr)
    os.remove(wavfilename)
    os.remove(mp3filename)
    x = np.reshape(x, X.shape)
    Y = torch.from_numpy(x).to(X) - X
    return Y[:, 0:orig_T]

def get_chroma_filterbank(sr, win, o1=-4, o2=4):
    """
    Compute a chroma matrix
    
    Parameters
    ----------
    sr: int
        Sample rate
    win: int
        STFT Window length
    o1: int
        Octave to start
    o2: int
        Octave to end
    
    Returns
    -------
    tensor(floor(win/2)+1, 12)
        A matrix, where each row has a bunch of Gaussian blobs
        around the center frequency of the corresponding note over
        all of its octaves
    """
    K = win//2+1 # Number of non-redundant frequency bins
    C = torch.zeros((K, 12)) # Create the matrix
    freqs = sr*torch.arange(K)/win # Compute the frequencies at each spectrogram bin
    for p in range(12):
        for octave in range(o1, o2+1):
            fc = 440*2**(p/12 + octave)
            sigma = 0.02*fc
            bump = torch.exp(-(freqs-fc)**2/(2*sigma**2))
            C[:, p] += bump
    return C

def get_batch_chroma(X, win_length, hop_length, hann, chroma_filterbank):
    """
    Compute the chroma on a batch of audio samples

    Parameters
    ----------
    X: torch.tensor(n_batches, time_samples)
        Batches of audio samples
    win_length: int
        Window length
    hop_length: int
        Hop length
    hann: torch.tensor(win_length)
        Hann window
    chroma_filterbank: torch.tensor(floor(win_length/2)+1, 12)
        Chroma filterbank
    
    Returns
    -------
    torch.tensor(n_batches, (time_samples-win_length)//hop_length+1, 12)
        Chroma for audio batches
    """
    S = torch.abs(torch.stft(X, win_length, hop_length, win_length, hann, return_complex=True, center=False))
    C = torch.einsum('ijk, jl -> ilk', S, chroma_filterbank)
    return C.swapaxes(1, 2)