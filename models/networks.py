import numpy as np
import torch
from torch import nn
import sys
sys.path.append("../src")
from audioutils import get_filtered_noise

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
            
def modified_sigmoid(x):
    return 2*torch.sigmoid(x)**np.log(10) + 1e-7
    
        
class CurveEncoder(nn.Module):
    def __init__(self, mlp_depth, n_units, n_taps, win_length, pre_scale=0.01):
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
        """
        super(CurveEncoder, self).__init__()
        self.win_length = win_length
        self.hop_length = win_length//2
        self.pre_scale = pre_scale
        
        self.YMLP = MLP(mlp_depth, 5, n_units) # Curve MLP
        self.LMLP = MLP(mlp_depth, 1, n_units) # Loudness MLP
        
        self.gru = nn.GRU(input_size=n_units*2, hidden_size=n_units, num_layers=1, bias=True, batch_first=True)
        self.FinalMLP = MLP(mlp_depth, n_units*3, n_units)
        self.TapsDecoder = nn.Linear(n_units, n_taps)
        self.AmplitudeDecoder = nn.Linear(n_units, 1)
        
    
    def forward(self, Y, L):
        """
        Parameters
        ----------
        Y: torch.tensor(n_batches, T, 5)
            xyrgb samples
        L: torch.tensor(n_batches, T, 1)
            
        """
        YOut = self.YMLP(Y)
        LOut = self.LMLP(L)
        YL = torch.concatenate((YOut, LOut), axis=2)
        G = self.gru(YL)[0]
        G = torch.concatenate((YOut, LOut, G), axis=2)
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
        

        
class CurveDecoder(nn.Module):
    def __init__(self, mlp_depth, n_units, win_length):
        """
        Parameters
        ----------
        mlp_depth: int
            Depth of each multilayer perceptron
        n_units: int
            Number of units in each multilayer perceptron
        win_length: int
            Length of window for each windowed audio chunk
        """
        super(CurveDecoder, self).__init__()
        self.win_length = win_length
        
        self.SMLP = MLP(mlp_depth, win_length//2+1, n_units) # STFT MLP
        
        self.gru = nn.GRU(input_size=n_units, hidden_size=n_units, num_layers=1, bias=True, batch_first=True)
        self.FinalMLP = MLP(mlp_depth, n_units*2, n_units)
        self.YDecoder = nn.Linear(n_units, 5)
        
    
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
        return modified_sigmoid(self.YDecoder(final))
    
    def get_num_parameters(self):
        total = 0
        for p in self.parameters():
            total += np.prod(p.shape)
        return total
