import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import librosa
import os
import subprocess
import sys
import glob
import time

if __name__ == '__main__':
    ## Step 1: Establish repository path and import source files
    repo_path = sys.argv[0]
    repo_path = repo_path.split("stftpowerpcaexperiment.py")[0] + ".."
    sys.path.append(repo_path + "/src")
    from spectrogramtools import *
    from stego import *
    from swpowerstego import *
    from curvature import *
    
    
    ## Step 2: Load audio
    audio_idx = int(sys.argv[1])
    audio_files = sorted(glob.glob("{}/data/GTzan/*/*.au".format(repo_path)))
    n_audiofiles = len(audio_files)
    audio_filename = audio_files[audio_idx]
    x, sr = librosa.load(audio_filename, sr=44100)
    win_length = 1024
    min_freq = 100
    max_freq = 8000

    ## Step 3: Find associated TSP paths in shuffled list
    np.random.seed(0)
    tsp_paths = glob.glob("{}/data/caltech-101/*/*.mat".format(repo_path))
    tsp_paths = sorted(tsp_paths)
    tsp_paths = [tsp_paths[i] for i in np.random.permutation(len(tsp_paths))]
    K = len(tsp_paths)//n_audiofiles
    tsp_paths = tsp_paths[K*audio_idx:K*(audio_idx+1)]

    ## Step 4: Solve for embedding
    fout = open("{}/results/PCA/{}.txt".format(repo_path, audio_idx), "w")
    for tsp_path in tsp_paths:
        X = sio.loadmat(tsp_path)["X"]
        # First store length of normalized TSP path
        fout.write("{} {}\n".format(tsp_path, get_length(X)))

        # Solve for fit
        for win in [100, 200, 400, 800]:
            for fit_lam in [0.001, 0.1, 1, 10]:
                for do_viterbi in [False, True]:
                    q = -1
                    sp = STFTPowerDisjointPCA(x, X, sr, win_length, min_freq, max_freq, win, fit_lam, q, do_viterbi=do_viterbi)
                    tic = time.time()
                    sp.solve(use_constraints=False)
                    elapsed = time.time()-tic
                    y = sp.reconstruct_signal()
                    snr_before = get_snr(x, y)
                    distortion_before = sp.get_distortion()[0]
                    Y = sp.get_signal(normalize=True)
                    length_before = get_length(Y)

                    prefix = "{}/results/{}_SpecPowerPCA_Win{}_fit{:.3g}_stftwin{}_freqs{}_{}_q{}".format(repo_path, audio_idx, win, fit_lam, win_length, min_freq, max_freq, q)
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
                    
                    z_sp = STFTPowerDisjointPCA(z, X, sr, win_length, min_freq, max_freq, win, fit_lam, q, do_viterbi=do_viterbi)
                    z_sp.targets = sp.targets
                    Y = z_sp.get_signal(normalize=True)
                    snr = get_snr(x, z)
                    distortion = z_sp.get_transformed_distortion()[1]
                    length = get_length(Y)
                    fout.write("{} {} {} {} {} {} {} {} {} {}\n".format(win, fit_lam, do_viterbi, snr_before, distortion_before, length_before, snr, distortion, length, elapsed))
                    fout.flush()
    fout.close()
