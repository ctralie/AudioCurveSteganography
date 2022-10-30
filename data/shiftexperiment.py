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
    repo_path = repo_path.split("shiftexperiment.py")[0] + ".."
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
    K = len(tsp_paths)//n_audiofiles
    tsp_paths = tsp_paths[K*audio_idx:K*(audio_idx+1)]

    win = 16
    do_viterbi=True
    q = -1
    n_trials = 4

    res = {'fit_lam':[], 'distortion':[], 'snr':[], 'length':[], 'tsp_path':[], 'audio_filename':[], 'gt_shift':[], 'recovered_shift':[], 'recovered_distortion':[]}
    for tsp_path in tsp_paths:
        ## Step 4: Do embedding and create audio clip
        X = sio.loadmat(tsp_path)["X"]
        s = get_arclen(get_curv_vectors(X, 0, 1, loop=True)[1])
        X = arclen_resample(X, s, X.shape[0])
        sigma = 1
        X = get_curv_vectors(X, 0, sigma, loop=True)[0]

        for fit_lam in [0.1, 1, 10]:
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
            snr = get_snr(x, z)
            distortion = z_sp.get_distortion()[0]
            length = get_length(Y)
            
            for trial in range(n_trials):
                res['tsp_path'].append(tsp_path)
                res['audio_filename'].append(audio_filename)
                res['fit_lam'].append(fit_lam)
                res['distortion'].append(distortion)
                res['snr'].append(snr)
                res['length'].append(length)
                # Choose a ground truth shift
                gt_shift = np.random.randint(win_length)
                res['gt_shift'].append(win_length-gt_shift)

                z2 = z[gt_shift::]
                Sz2 = stft_disjoint(z2, win_length)


                shifts = []
                lengths = []
                distortions = []
                for shift in range(win_length):
                    if shift > 0 and shift%100 == 0:
                        print(".", end='', flush=True)
                    shifts.append(shift)
                    z2s = z2[shift::]
                    Sz2 = stft_disjoint(z2s, win_length)
                    M = Sz2.shape[1]-win+1
                    cut = sp.MagSolver.targets[0].size - M
                    y_sp = STFTPowerDisjoint(z2s, X, win_length, freqs, freqs, win, fit_lam, q, do_viterbi=False)
                    lengths.append(get_length(y_sp.get_signal(normalize=True)))
                    y_sp.MagSolver.targets = [t[cut::] for t in sp.MagSolver.targets]
                    distortions.append(y_sp.get_distortion()[0])
                idx = int(np.argmin(lengths))
                res['recovered_shift'].append(idx)
                res['recovered_distortion'].append(distortions[idx])
                print("gt_shift", win_length-gt_shift, ", recovered_shift", idx)
                json.dump(res, open("{}/results/shift{}.json".format(repo_path, audio_idx), "w"))