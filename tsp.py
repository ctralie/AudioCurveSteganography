import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def sinebow(h):
    # https://twitter.com/jon_barron/status/1388233935641976833
    f = lambda x : np.sin(np.pi * x)**2
    return np.stack([f(3/6-h), f(5/6-h), f(7/6-h)], -1)

class UFFast:
    """
    Union find (helper data structure for Kruskal's)
    """
    def __init__(self, N):
        self.parent = [i for i in range(N)]
        self.size = [1 for i in range(1)]
        self._operations = 0
        self._calls = 0
    
    def root(self, i):
        p = i
        while self.parent[p] != p:
            p = self.parent[p]
            self._operations += 1
        new_parent = p
        p = i
        while self.parent[p] != p:
            self.parent[p] = new_parent
            p = self.parent[p]
            self._operations += 1
        return new_parent
    
    def find(self, i, j):
        self._calls += 1
        return self.root(i) == self.root(j)
    
    def union(self, i, j):
        self._calls += 1
        rooti = self.root(i)
        rootj = self.root(j)
        if rooti != rootj:
            self.parent[rooti] = rootj
            self._operations += 1

class GraphNode:
    def __init__(self):
        self.edges = []
        self.data = {}

def get_mst_kruskal(nodes, edges):
    """
    Compute the minimum spanning tree using Kruskal's algorihtm
    """
    edges = sorted(edges, key = lambda e: e[2])
    djset = UFFast(len(nodes))
    new_edges = []
    for e in edges:
        (i, j, d) = e
        if not djset.find(i, j):
            djset.union(i, j)
            new_edges.append(e)
            nodes[i].edges.append(nodes[j])
            nodes[j].edges.append(nodes[i])
    return new_edges

def make_delaunay_graph(X):
    from scipy.spatial import Delaunay
    N = X.shape[0]
    nodes = []
    for i in range(N):
        n = GraphNode()
        n.data = {'x':X[i, 0], 'y':X[i, 1]}
        nodes.append(n)
    tri = Delaunay(X).simplices
    edges = set()
    for i in range(tri.shape[0]):
        for k in range(3):
            i1, i2 = tri[i, k], tri[i, (k+1)%3]
            diff = X[i1, :] - X[i2, :]
            d = np.sqrt(np.sum(diff**2))
            edges.add((i1, i2, d))
    return nodes, list(edges)

def do_dfs(node, order):
    if not node.visited:
        order.append(node)
    node.visited = True
    for other in node.edges:
        if not other.visited:
            do_dfs(other, order)


@jit(nopython=True)
def get_improvement_indices(Y, i_last):
    """
    Return the indices of a 2-opt swap that will decrease
    the overall distance

    Parameters
    ----------
    Y: ndarray(N, 2)
        Current tour
    i_last: int
        Index of last element in the tour to have been swapped in 2-opt.
        Search is more likely to find a swap if we start here
    """
    diff = np.sum((Y[1::, :] - Y[0:-1, :])**2, axis=1)
    dists = np.sqrt(diff)
    N = Y.shape[0]
    for i in list(range(i_last, N-1)) + list(range(1, i_last)):
        for k in range(i+1, N-1):
            d1 = np.sqrt(np.sum((Y[i, :] - Y[k, :])**2))
            d2 = np.sqrt(np.sum((Y[i+1, :] - Y[k+1, :])**2))
            if d1 + d2 < dists[i] + dists[k]:
                return (i, k, dists[i]+dists[k], d1+d2)


def refine_tour(X, idxs, max_iters=10000, plot_interval=0):
    """
    Refine the TSP tour by doing a sequence of 2-opt moves

    Parameters
    ----------
    X: ndarray(N, 2)
        Input points
    idxs: ndarray(N+1)
        Indices of initial guess for tour, with first and last equal
    max_iters: int
        Maximum number of 2-opt iterations to do
    plot_interval: int
        If > 0, plot progress ever plot_interval frames
    
    Returns
    -------
    ndarray(N+1)
        Refined tour
    """
    if plot_interval > 0:
        plt.figure(figsize=(12, 12))
    i_last = 1
    idxs_ret = np.array(idxs, dtype=int)
    for it in range(max_iters):
        res = get_improvement_indices(X[idxs_ret, :], i_last)
        if not res:
            return idxs_ret
        i, k, db, da = res
        i_last = i
        idxs1 = idxs_ret[0:i+1]
        idxsmid = idxs_ret[i+1:k+1]
        idxs2 = idxs_ret[k+1::]
        idxs_ret = np.concatenate((idxs1, idxsmid[::-1], idxs2))
        Y = X[idxs_ret, :]
        diff = np.sum((Y[1::, :] - Y[0:-1, :])**2, axis=1)
        dist = np.sum(np.sqrt(diff))
        if plot_interval > 0 and it%plot_interval == 0:
            plt.clf()
            plt.scatter(Y[:, 0], Y[:, 1], s=15, c=sinebow(np.linspace(0, 1, Y.shape[0])))
            plt.plot(Y[:, 0], Y[:, 1], 'k')
            plt.gca().set_facecolor((0.8, 0.8, 0.8))
            plt.axis("equal")
            plt.title("Swapping {} and {}, Distance: {:.6f}, Distance Before: {:.3f}, Distance After: {:.3f}".format(i, k, dist, db, da))
            plt.savefig("TSPIter{}.png".format(it), facecolor='white')
    return idxs

def get_tsp_tour(X, max_iters=10000, plot_interval=0):
    """
    Compute an approximate traveling salesperson tour by first performing a
    depth first search on a Euclidean MST, followed by 2-opt swaps

    Parameters
    ----------
    X: ndarray(N, 2)
        Input points
    max_iters: int
        Maximum number of 2-opt iterations to do
    plot_interval: int
        If > 0, plot progress ever plot_interval frames
    
    Returns
    -------
    ndarray(N, 2)
        TSP Tour
    """
    nodes, edges = make_delaunay_graph(X)
    get_mst_kruskal(nodes, edges)
    order = []
    for i, node in enumerate(nodes):
        node.visited = False
        node.i = i
    do_dfs(nodes[0], order)
    order.append(order[0])
    X = np.array([[n.data['x'], n.data['y']] for n in nodes])
    idxs = [n.i for n in order]
    idxs = refine_tour(X, idxs, max_iters=max_iters, plot_interval=plot_interval)
    return X[idxs[0:-1], :]

def density_filter(X, fac, k=1):
    """
    Filter out points below a certain density

    Parameters
    ----------
    X: ndarray(N, 2)
        Point cloud
    fac: float
        Percentile (between 0 and 1) of points to keep, by density
    k: int
        How many neighbors to consider
    
    Returns
    -------
    ndarray(N)
        Distance of nearest point
    """
    from scipy.spatial import KDTree
    tree = KDTree(X)
    dd, _ = tree.query(X, k=k+1)
    dd = np.mean(dd[:, 1::], axis=1)
    q = np.quantile(dd, fac)
    return X[dd < q, :]
