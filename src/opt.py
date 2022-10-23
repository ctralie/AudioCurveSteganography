import numpy as np

def lsq_box(A, b, x0, xmin, xmax, alpha, n_iters):
    x = np.array(x0)
    resid = []
    for it in range(n_iters):
        res = A.dot(x) - b
        resid.append(np.sum(res**2))
        x = x - alpha*(A.T).dot(res)
        x = np.maximum(np.minimum(x, xmax), xmin)
        if it%(n_iters//50 == 0):
            print(it, resid[-1])
    return x, resid