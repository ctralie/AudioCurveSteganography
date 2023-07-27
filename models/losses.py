import torch

HANN_TABLE = {}

def mss_loss(X, Y, eps=1e-7):
    """
    Compute the multi-scale spectral loss between two tensors

    Parameters
    ----------
    X: torch.tensor(n_batches, n_samples, 1)
        First batch of audio
    Y: torch.tensor(n_batches, n_samples, 1)
        Second batch of audio
    
    Returns
    -------
    Differentiable torch float: MSS Loss
    """
    loss = 0
    win = 64
    while win <= 2048:
        hop = win//4
        if not win in HANN_TABLE:
            HANN_TABLE[win] = torch.hann_window(win).to(X)
        hann = HANN_TABLE[win]
        SX = torch.abs(torch.stft(X.squeeze(), win, hop, win, hann, return_complex=True))
        SY = torch.abs(torch.stft(Y.squeeze(), win, hop, win, hann, return_complex=True))
        loss_win = torch.sum(torch.abs(SX-SY)) + torch.sum(torch.abs(torch.log(SX+eps)-torch.log(SY+eps)))
        loss += loss_win/torch.numel(SX)
        win *= 2
    return loss