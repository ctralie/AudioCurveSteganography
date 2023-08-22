from unicodedata import bidirectional
import numpy as np
import torch
from torch import nn
import sys
sys.path.append("../src")

class MLP(nn.Module):
    def __init__(self, depth=3, n_input=1, n_units=512):
        super(MLP, self).__init__()
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(n_input, n_units))
            else:
                layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.LayerNorm(normalized_shape=n_units))
            layers.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
    def get_num_parameters(self):
        total = 0
        for p in self.parameters():
            total += np.prod(p.shape)
        return total

def get_binary_encoding(Y, bits_per_channel=8):
    Y = (2**bits_per_channel)*Y
    YBin = torch.zeros(Y.shape[0], Y.shape[1], Y.shape[2]*bits_per_channel)
    YBin = YBin.to(Y)
    place = 2**(bits_per_channel-1)
    idx = 0
    while place > 0:
        bit = 1.0*(Y-place > 0)
        YBin[:, :, idx*Y.shape[2]:(idx+1)*Y.shape[2]] = bit
        Y -= bit*place
        place = place // 2
        idx += 1
    return YBin

def decode_binary(YBin, bits_per_channel=8):
    dim = YBin.shape[2]//bits_per_channel
    Y = torch.zeros(YBin.shape[0], YBin.shape[1], dim)
    Y = Y.to(YBin)
    place = 2**(bits_per_channel-1)
    idx = 0
    while place > 0:
        Y += place*YBin[:, :, idx*dim:(idx+1)*dim]
        place = place // 2
        idx += 1
    return Y/(2**bits_per_channel-1)

        
class CurveEncoder(nn.Module):
    def __init__(self, mlp_depth, n_units, n_taps, win_length, pre_scale=0.01, dim=5):
        """
        Parameters
        ----------
        mlp_depth: int
            Depth of each multilayer perceptron
        n_units: int
            Number of units in each multilayer perceptron
        n_taps: int
            Number of taps in each FIR filter
        win_length: int
            Length of window for each windowed audio chunk
        pre_scale: float
            Initial ampitude of noise (try to start off much lower than audio)
        dim: int
            Number of dimensions in the curve
        """
        super(CurveEncoder, self).__init__()
        self.win_length = win_length
        self.hop_length = win_length//2
        self.pre_scale = pre_scale
        
        self.YMLP = MLP(mlp_depth, dim, n_units) # Curve MLP
        self.LMLP = MLP(mlp_depth, 1, n_units) # Loudness MLP
        self.CMLP = MLP(mlp_depth, 12, n_units) # Chroma MLP
        
        self.gru = nn.GRU(input_size=n_units*3, hidden_size=n_units, num_layers=1, bias=True, batch_first=True)
        self.FinalMLP = MLP(mlp_depth, n_units*4, n_units)
        self.TapsDecoder = nn.Linear(n_units, n_taps)
        self.AmplitudeDecoder = nn.Linear(n_units, 1)
        
    
    def forward(self, Y, L, C):
        """
        Parameters
        ----------
        Y: torch.tensor(n_batches, T, 5)
            xyrgb samples
        L: torch.tensor(n_batches, T, 1)
            Loudness samples
        C: torch.tensor(n_batches, T, 12)
            Chroma samples
        """
        from audioutils import get_filtered_noise
        YOut = self.YMLP(Y)
        LOut = self.LMLP(L)
        COut = self.CMLP(C)
        YLC = torch.concatenate((YOut, LOut, COut), axis=2)
        G = self.gru(YLC)[0]
        G = torch.concatenate((YOut, LOut, COut, G), axis=2)
        final = self.FinalMLP(G)
        H = nn.functional.tanh(self.TapsDecoder(final))
        A = nn.functional.leaky_relu(self.AmplitudeDecoder(final))
        N = get_filtered_noise(H, A, self.win_length)
        return self.pre_scale*N
    
    def get_num_parameters(self):
        total = 0
        for p in self.parameters():
            total += np.prod(p.shape)
        return total
        

        
class CurveSTFTEncoder(nn.Module):
    def __init__(self, mlp_depth, n_units, win_length, n_taps, max_lag, tap_amp, tap_sigma, dim=5):
        """
        Parameters
        ----------
        mlp_depth: int
            Depth of each multilayer perceptron
        n_units: int
            Number of units in each multilayer perceptron
        win_length: int
            Length of window for each windowed audio chunk
        n_taps: int
            Number of taps in each echo filter
        max_lag: int
            Maximum lag of tap
        tap_amp: float
            Maximum amplitude of taps
        tap_sigma: float
            Maximum sigma of taps
        dim: int
            Number of dimensions in the curve
        """
        super(CurveSTFTEncoder, self).__init__()
        self.win_length = win_length
        self.hop_length = win_length//2
        self.max_lag = max_lag
        self.tap_amp = tap_amp
        self.tap_sigma = tap_sigma
        
        self.YMLP = MLP(mlp_depth, dim, n_units) # Curve MLP
        self.SMLP = MLP(mlp_depth, win_length//2+1, n_units) # STFT MLP
        
        self.gru = nn.GRU(input_size=n_units*2, hidden_size=n_units, num_layers=1, bias=True, batch_first=True)
        self.FinalMLP = MLP(mlp_depth, n_units*3, n_units)
        self.AmpDecoder = nn.Linear(n_units, n_taps)
        self.LocDecoder = nn.Linear(n_units, n_taps)
        self.SigmaDecoder = nn.Linear(n_units, n_taps)
        
    
    def forward(self, X, Y):
        """
        Parameters
        ----------
        X: torch.tensor(n_batches, (T-1)*hop_length*win_length)
            Audio Samples
        Y: torch.tensor(n_batches, T, 5)
            xyrgb samples
        """
        from audioutils import get_zerophase_filtered_signals
        hann = torch.hann_window(self.win_length).to(X)
        win_length = self.win_length
        hop_length = self.hop_length
        S = torch.stft(X, win_length, hop_length, win_length, hann, return_complex=True, center=True)
        S = torch.abs(S)

        YOut = self.YMLP(Y)
        SOut = self.SMLP(S.swapaxes(1, 2)[:, 0:Y.shape[1], :])
        YS = torch.concatenate((YOut, SOut), axis=2)
        G = self.gru(YS)[0]
        G = torch.concatenate((YOut, SOut, G), axis=2)
        final = self.FinalMLP(G)

        A = self.tap_amp*torch.tanh(self.AmpDecoder(final))
        Loc = self.max_lag*torch.sigmoid(self.LocDecoder(final))
        Sigma = self.tap_sigma*torch.sigmoid(self.SigmaDecoder(final))
        shape = (Loc.shape[0], Loc.shape[1], Loc.shape[2], 1)
        t = torch.arange(self.max_lag+1).view(1, 1, 1, self.max_lag+1).to(Loc)
        taps = A.view(shape)*torch.exp(-(Loc.view(shape)-t)**2/(2*Sigma.view(shape)**2))
        taps = torch.sum(taps, dim=2)
        taps[:, :, 0] = 1

        return get_zerophase_filtered_signals(taps, X, 1, win_length, renorm_amp=False), taps

    
    def get_num_parameters(self):
        total = 0
        for p in self.parameters():
            total += np.prod(p.shape)
        return total
        


class CurveDecoder(nn.Module):
    def __init__(self, mlp_depth, n_units, win_length, dim):
        """
        Parameters
        ----------
        mlp_depth: int
            Depth of each multilayer perceptron
        n_units: int
            Number of units in each multilayer perceptron
        win_length: int
            Length of window for each windowed audio chunk
        dim: int
            Number of dimensions in the curve
        """
        super(CurveDecoder, self).__init__()
        self.win_length = win_length
        self.dim = dim
        
        self.SMLP = MLP(mlp_depth, win_length//2+1, n_units) # STFT MLP
        
        self.gru = nn.GRU(input_size=n_units, hidden_size=n_units, num_layers=1, bias=True, batch_first=True)
        self.FinalMLP = MLP(mlp_depth, n_units*2, n_units)
        self.YDecoder = nn.Linear(n_units, dim)
        
    
    def forward(self, X):
        """
        Parameters
        ----------
        Y: torch.tensor(n_batches, n_samples)
            Audio samples
        """
        win = self.win_length
        hop = win//2
        hann = torch.hann_window(win).to(X)
        S = torch.abs(torch.stft(X, win, hop, win, hann, return_complex=True, center=False))
        S = torch.swapaxes(S, 1, 2)
        SOut = self.SMLP(S)
        G = self.gru(SOut)[0]
        G = torch.concatenate((SOut, G), axis=2)
        final = self.FinalMLP(G)
        return torch.sigmoid(self.YDecoder(final))
    
    def get_num_parameters(self):
        total = 0
        for p in self.parameters():
            total += np.prod(p.shape)
        return total


        
class BinaryDecoder(nn.Module):
    def __init__(self, mlp_depth, n_units, win_length, n_bits):
        """
        Parameters
        ----------
        mlp_depth: int
            Depth of each multilayer perceptron
        n_units: int
            Number of units in each multilayer perceptron
        win_length: int
            Length of window for each windowed audio chunk
        n_bits: int
            Number of bits to decode
        """
        super(BinaryDecoder, self).__init__()
        self.win_length = win_length
        self.n_bits = n_bits
        
        self.SMLP = MLP(mlp_depth, win_length//4, n_units) # STFT MLP
        
        self.gru = nn.GRU(input_size=n_units, hidden_size=n_units, num_layers=1, bias=True, batch_first=True)
        self.FinalMLP = MLP(mlp_depth, n_units*2, n_units)
        self.LogBitDecoder = nn.Linear(n_units, n_bits, bias=False)
        
    
    def forward(self, X):
        """
        Parameters
        ----------
        X: torch.tensor(n_batches, n_samples)
            Audio samples
        """
        win = self.win_length
        hop = win//2
        hann = torch.hann_window(win).to(X)
        S = torch.abs(torch.stft(X, win, hop, win, hann, return_complex=True, center=False))
        S = torch.swapaxes(S, 1, 2)[:, :, 1:win//4+1]
        SOut = self.SMLP(S)
        G = self.gru(SOut)[0]
        G = torch.concatenate((SOut, G), axis=2)
        final = self.FinalMLP(G)
        return self.LogBitDecoder(final)
    
    def get_num_parameters(self):
        total = 0
        for p in self.parameters():
            total += np.prod(p.shape)
        return total
