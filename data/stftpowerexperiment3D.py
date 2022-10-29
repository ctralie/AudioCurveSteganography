import numpy as np
from scipy.io import wavfile
import librosa
import os
import subprocess
import sys
import glob
import time

def get_length(X):
    """
    Compute the length along a Euclidean curve

    Parameters
    ----------
    X: ndarray(N, d)
        Input curve
    
    Returns
    -------
    float: Length along X
    """
    Y = np.array(X)
    Y -= np.min(Y, axis=0)[None, :]
    Y /= np.max(Y, axis=0)[None, :]
    d = Y[1::, :] - Y[0:-1, :]
    d = np.sqrt(np.sum(d**2, axis=1))
    return np.sum(d)

if __name__ == '__main__':
    ## Step 1: Establish repository path and import source files
    repo_path = sys.argv[0]
    repo_path = repo_path.split("stftpowerexperiment3D.py")[0] + ".."
    sys.path.append(repo_path + "/src")
    from spectrogramtools import *
    from stego import *
    from swpowerstego import *
    from curvature import *
    from trimesh import *
    
    
    ## Step 2: Load audio
    audio_idx = int(sys.argv[1])
    audio_files = sorted(glob.glob("{}/data/GTzan/*/*.au".format(repo_path)))
    n_audiofiles = len(audio_files)
    audio_filename = audio_files[audio_idx]
    x, sr = librosa.load(audio_filename, sr=44100)
    win_length = 1024
    freqs = [1, 2, 3]

    ## Step 3: Find associated TSP paths in shuffled list
    off_files = glob.glob("{}/data/meshseg/*.off".format(repo_path))
    off_files = sorted(off_files)
    K = 4
    off_files *= int(np.ceil(n_audiofiles*K/len(off_files)))
    np.random.seed(0)
    off_files = [off_files[i] for i in np.random.permutation(len(off_files))]
    off_files = off_files[K*audio_idx:K*(audio_idx+1)]

    ## Step 4: Solve for embedding
    fout = open("{}/results/3D_{}.txt".format(repo_path, audio_idx), "w")
    for off_file in off_files:
        VPos, _, ITris = load_off(off_file)
        hamprefix = "{}/{}".format(repo_path, audio_idx)
        X = get_hamiltonian_cycle(VPos, ITris, prefix=hamprefix, blossom_path="{}/blossom5".format(repo_path))
        s = get_arclen(get_curv_vectors(X, 0, 1, loop=True)[1])
        X = arclen_resample(X, s, X.shape[0])
        sigma = 1
        X = get_curv_vectors(X, 0, sigma, loop=True)[0]

        # First store length of normalized TSP path
        fout.write("{} {}\n".format(off_file, get_length(X)))

        # Solve for fit
        for win in [1, 2, 4, 8, 16, 32, 64]:
            for fit_lam in [0.1, 1, 10]:
                for do_viterbi in [False, True]:
                    q = -1
                    sp = STFTPowerDisjoint(x, X, win_length, freqs, freqs, win, fit_lam, q, do_viterbi=do_viterbi)
                    tic = time.time()
                    sp.solve()
                    elapsed = time.time()-tic
                    y = sp.reconstruct_signal()
                    snr_before = get_snr(x, y)
                    distortion_before = sp.get_distortion()[0]
                    Y = sp.get_signal(normalize=True)
                    length_before = get_length(Y)

                    prefix = "{}/results/{}_SpecPower_Win{}_fit{:.3g}_stftwin{}_freqs{}_{}_q{}".format(repo_path, audio_idx, win, fit_lam, win_length, freqs[0], freqs[1], q)
                    mp3filename = "{}.mp3".format(prefix)
                    wavfilename = "{}.wav".format(prefix)
                    wavfile.write(wavfilename, sr, y)
                    # Step 1: Convert from wav to MP3
                    if os.path.exists(mp3filename):
                        os.remove(mp3filename)
                    subprocess.call(["{}/ffmpeg".format(repo_path), "-i", wavfilename, mp3filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    os.remove(wavfilename)

                    # Step 2: Convert from mp3 back to wav so it can be read
                    subprocess.call(["{}/ffmpeg".format(repo_path), "-i", mp3filename, wavfilename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    z, sr = librosa.load(wavfilename, sr=sr)
                    os.remove(wavfilename)
                    os.remove(mp3filename)
                    
                    z_sp = STFTPowerDisjoint(z, X, win_length, freqs, freqs, win, fit_lam, q, do_viterbi=do_viterbi)
                    z_sp.MagSolver.targets = sp.MagSolver.targets
                    Y = z_sp.get_signal(normalize=True)
                    snr = get_snr(x, z)
                    distortion = z_sp.get_distortion()[0]
                    length = get_length(Y)
                    fout.write("{} {} {} {} {} {} {} {} {} {}\n".format(win, fit_lam, do_viterbi, snr_before, distortion_before, length_before, snr, distortion, length, elapsed))
                    fout.flush()
    fout.close()