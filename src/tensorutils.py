import torch
from torch import nn

def upsample_time(X, hop_length, mode='nearest'):
    """
    Upsample a tensor by a factor of hop_length along the time axis
    
    Parameters
    ----------
    X: torch.tensor(M, T, N)
        A tensor in which the time axis is axis 1
    hop_length: int
        Upsample factor
    mode: string
        Mode of interpolation.  'nearest' by default to avoid artifacts
        where notes in the violin jump by large intervals
    
    Returns
    -------
    torch.tensor(M, T*hop_length, N)
        Upsampled tensor
    """
    X = X.permute(0, 2, 1)
    X = nn.functional.interpolate(X, size=hop_length*X.shape[-1], mode=mode)
    return X.permute(0, 2, 1)
