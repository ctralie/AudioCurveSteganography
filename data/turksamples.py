import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import librosa
import os
import subprocess
import sys
import glob
import time
import json

if __name__ == '__main__':
    ## Step 1: Establish repository path and import source files
    repo_path = sys.argv[0]
    repo_path = repo_path.split("turksamples.py")[0] + ".."
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
    freqs = [1, 2]

    ## Step 3: Find associated TSP paths in shuffled list
    np.random.seed(0)
    tsp_paths = glob.glob("{}/data/caltech-101/*/*.mat".format(repo_path))
    tsp_paths = sorted(tsp_paths)
    tsp_paths = [tsp_paths[i] for i in np.random.permutation(len(tsp_paths))]
    tsp_path = tsp_paths[audio_idx]
    print(tsp_path)

    ## Step 4: Do embedding and create audio clip
    X = sio.loadmat(tsp_path)["X"]
    s = get_arclen(get_curv_vectors(X, 0, 1, loop=True)[1])
    X = arclen_resample(X, s, X.shape[0])
    sigma = 1
    X = get_curv_vectors(X, 0, sigma, loop=True)[0]

    # First store length of normalized TSP path
    win = 16
    do_viterbi=True
    fit_lam = [0.1, 1, 10, np.inf][audio_idx%4]
    distortion = np.inf
    snr = np.inf
    length = np.inf
    turk_prefix = "{}/data/turksamples/{}".format(repo_path, audio_idx)

    y = np.array(x)
    if np.isfinite(fit_lam):
        q = -1
        sp = STFTPowerDisjoint(x, X, win_length, freqs, freqs, win, fit_lam, q, do_viterbi=do_viterbi)
        tic = time.time()
        sp.solve()
        elapsed = time.time()-tic
        y = sp.reconstruct_signal()
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
        YGT = z_sp.get_target(normalize=True)
        snr = get_snr(x, z)
        distortion = z_sp.get_distortion()[0]
        length = get_length(Y)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(YGT[:, 0], YGT[:, 1])
        plt.axis("equal")
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title("Ground Truth")
        plt.subplot(122)
        plt.plot(Y[:, 0], Y[:, 1])
        plt.axis("equal")
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title("Reconstructed")
        plt.savefig("{}.png".format(turk_prefix))

    # Pick a random 10 seconds
    idx = np.random.randint(x.size-10*sr)
    y = y[idx:idx+10*sr]
    mp3filename = "{}.mp3".format(turk_prefix)
    wavfilename = "{}.wav".format(turk_prefix)
    wavfile.write(wavfilename, sr, y)
    subprocess.call(["{}/ffmpeg".format(repo_path), "-i", wavfilename, mp3filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    mp3size = os.path.getsize(mp3filename)
    wavsize = os.path.getsize(wavfilename)
    compression_ratio = wavsize/mp3size
    print("Compression ratio", compression_ratio)
    os.remove(wavfilename)

    res = {'fit_lam':fit_lam, 'distortion':distortion, 'snr':snr, 'length':length, 'compression_ratio':compression_ratio}
    json.dump(res, open("{}/data/turksamples/{}.json".format(repo_path, audio_idx), "w"))