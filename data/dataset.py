import glob
import pickle
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("../src")
from audioutils import extract_loudness

class CurveData(Dataset):
    def __init__(self, rg, tour_samples, T, samples_per_batch, voronoi=True):
        """
        Parameters
        ----------
        rg: list(int)
            List of indices to take in each image class (used for test/train split)
        tour_samples: int
            Number of samples in each TSP tour
        T: int
            Number of samples to take in each chunk
        samples_per_batch: int
            Number of samples per batch
        voronoi: bool
            If True, use voronoi images.  If False, use wavelet images
        """
        self.files = []
        for c in glob.glob("../data/imagenet/*"): # Go through each class
            files = glob.glob("{}/*.pkl".format(c))
            if voronoi:
                files = [f for f in files if not "wavelet" in f]
            else:
                files = [f for f in files if "wavelet" in f]
            files = sorted(files)
            self.files += [files[i] for i in rg]
        # Load in all curves
        self.Ys = []
        for file in self.files:
            res = pickle.load(open(file, "rb"))
            Y = res[tour_samples]["Y"]
            if Y.shape[0] >= T:
                self.Ys.append(np.array(Y, dtype=np.float32))
        self.T = T
        self.samples_per_batch = samples_per_batch
    
    def __len__(self):
        return self.samples_per_batch
    
    def __getitem__(self, idx):
        """
        Pull out a random chunk of the appropriate length from a random file
        """
        idx = np.random.randint(len(self.Ys))
        Y = self.Ys[idx]
        Y = np.roll(Y, np.random.randint(Y.shape[0]), axis=0)
        Y = Y[0:self.T, :]
        return torch.from_numpy(Y)
        
class AudioData(Dataset):
    def __init__(self, file_pattern, T, samples_per_batch, win_length, sr):
        """
        Parameters
        ----------
        file_pattern: string
            File pattern to match for audio files
        T: int
            Number of windows to take in each chunk
        samples_per_batch: int
            Number of samples per batch
        win_length: int
            Window length of STFT; hop_length assumed to be half of this
        sr: int
            Sample rate
        """
        self.samples_per_batch = samples_per_batch
        hop_length = win_length//2
        self.hop_length = hop_length
        self.T = T
        self.n_samples = hop_length*(T-1)+win_length
        self.samples = []
        self.loudnesses = []
        self.chromas = []
        for filename in glob.glob(file_pattern):
            x, _ = librosa.load(filename, sr=sr)
            loudness = extract_loudness(x, sr, hop_length, n_fft=win_length)
            x = np.array(x, dtype=np.float32)
            loudness = np.array(loudness, dtype=np.float32)
            self.samples.append(x)
            self.loudnesses.append(loudness)

    def __len__(self):
        return self.samples_per_batch
    
    def __getitem__(self, idx):
        """
        Return a random audio clip, along with its loudness
        
        Returns
        -------
        torch.tensor(n_samples)
            Audio clip
        torch.tensor(T, 1)
            Loudness
        """
        idx = np.random.randint(len(self.samples))
        x = self.samples[idx]
        loudness = self.loudnesses[idx]
        i1 = np.random.randint(len(loudness)-self.T-1)
        loudness = loudness[i1:i1+self.T]
        loudness = loudness[:, None]
        i1 = i1*self.hop_length
        x = x[i1:i1+self.n_samples]
        return torch.from_numpy(x), torch.from_numpy(loudness)