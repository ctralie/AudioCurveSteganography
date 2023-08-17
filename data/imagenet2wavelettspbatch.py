
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import time
import torch
import glob
import sys
import pickle
import os
from skimage.transform import resize

if __name__ == '__main__':
    ## Step 1: Establish repository path and import source files
    repo_path = sys.argv[0]
    repo_path = repo_path.split("imagenet2wavelettspbatch.py")[0] + ".."
    sys.path.append(repo_path + "/src")
    from wavelets2d import get_color_wavelet_tsp, invert_sparse_coefficients

    ## Step 2: Figure out which files are involved
    files = glob.glob("{}/data/imagenet/*/*.JPEG".format(repo_path))
    files = sorted(files)
    num_batches = int(sys.argv[1])
    idx = int(sys.argv[2])
    K = int(np.ceil(len(files)/num_batches))
    files = files[idx*K:(idx+1)*K]

    ## Step 3: Do the TSP wavelet images on all of the files
    n_levels = 5
    box_scale = 50
    img_res = 256
    plt.figure(figsize=(8, 4))
    for f in files:
        fout = f.split(".JPEG")[0] + "_wavelet.pkl"
        if not os.path.exists(fout):
            print("Doing ", f)
            res = {}
            I = skimage.io.imread(f)
            if len(I.shape) == 2:
                Ik = I[:, :, None]
                I = np.concatenate((Ik, Ik, Ik), axis=2)
            # Crop to square image
            N = min(I.shape[0], I.shape[1])
            I = I[0:N, 0:N, ...]
            I = resize(I, (img_res, img_res), anti_aliasing=True)

            for n_points in [1000, 2000, 3000, 4000, 5000]:
                ## Get wavelet tour
                Y = get_color_wavelet_tsp(I, n_levels, n_points, box_scale)
                J = invert_sparse_coefficients(Y, I.shape[0], n_levels, box_scale)
                rmse = np.sqrt(np.sum((I-J)**2)/I.size)
                res[n_points] = {"rmse":rmse, "Y":Y}
                J[J < 0] = 0
                J[J > 1] = 1
                plt.clf()
                plt.subplot(121)
                plt.plot(Y[:, 0], Y[:, 1])
                plt.gca().invert_yaxis()
                plt.axis("equal")
                plt.subplot(122)
                plt.imshow(J)
                plt.title("{} Points, rmse = {:.3f}".format(Y.shape[0], rmse*255))
                plt.savefig(f.split(".JPEG")[0]+"_wavelet{}.png".format(n_points), bbox_inches='tight')
            pickle.dump(res, open(fout, "wb"))
