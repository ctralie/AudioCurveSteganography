
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import time
import torch
import glob
import sys
import pickle
import os

if __name__ == '__main__':
    ## Step 1: Establish repository path and import source files
    repo_path = sys.argv[0]
    repo_path = repo_path.split("imagenet2tspbatch.py")[0] + ".."
    sys.path.append(repo_path + "/src")
    from imtools import get_voronoi_image
    from tsp import get_tsp_tour

    ## Step 2: Figure out which files are involved
    files = glob.glob("{}/data/imagenet/*/*.JPEG".format(repo_path))
    files = sorted(files)
    num_batches = int(sys.argv[1])
    idx = int(sys.argv[2])
    K = int(np.ceil(len(files)/num_batches))
    files = files[idx*K:(idx+1)*K]

    ## Step 3: Do the TSP voronoi images on all of the files
    n_iters = 100
    device = 'cpu'
    plt.figure(figsize=(8, 4))
    for f in files:
        fout = f.split(".JPEG")[0] + ".pkl"
        if not os.path.exists(fout):
            print("Doing ", f)
            res = {}
            I = skimage.io.imread(f)
            # Crop to square image
            N = min(I.shape[0], I.shape[1])
            I = I[0:N, 0:N, ...]
            for n_points in [1000, 2000, 3000, 4000, 5000]:
                ## Get voronoi image
                J, X, final_cost = get_voronoi_image(I, device, n_points, n_neighbs=2, n_iters=n_iters, verbose=False, plot_iter_interval=0, use_lsqr=False)
                rmse = np.sqrt(final_cost/I.size)
                ## Do TSP
                Y = get_tsp_tour(X)
                res[n_points] = {"rmse":rmse, "Y":Y}
                plt.clf()
                plt.subplot(121)
                plt.scatter(Y[:, 0], Y[:, 1], c=Y[:, 2::])
                plt.gca().invert_yaxis()
                plt.axis("equal")
                plt.subplot(122)
                plt.imshow(J)
                plt.title("{} Points, rmse = {:.3f}".format(Y.shape[0], rmse*255))
                plt.savefig(f.split(".JPEG")[0]+"_voronoi{}.png".format(n_points), bbox_inches='tight')
            pickle.dump(res, open(fout, "wb"))
