import numpy as np 
from scipy import sparse 


def load_off(filename):
    """
    Load in an OFF file, assuming it's a triangle mesh
    Parameters
    ----------
    filename: string
        Path to file
    Returns
    -------
    VPos : ndarray (N, 3)
        Array of points in 3D
    VColors : ndarray(N, 3)
        Array of RGB colors
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    """
    fin = open(filename, 'r')
    nVertices = 0
    nFaces = 0
    lineCount = 0
    face = 0
    vertex = 0
    divideColor = False
    VPos = np.zeros((0, 3))
    VColors = np.zeros((0, 3))
    ITris = np.zeros((0, 3))
    for line in fin:
        lineCount = lineCount+1
        fields = line.split() #Splits whitespace by default
        if len(fields) == 0: #Blank line
            continue
        if fields[0][0] in ['#', '\0', ' '] or len(fields[0]) == 0:
            continue
        #Check section
        if nVertices == 0:
            if fields[0] == "OFF" or fields[0] == "COFF":
                if len(fields) > 2:
                    fields[1:4] = [int(field) for field in fields]
                    [nVertices, nFaces, nEdges] = fields[1:4]
                    #Pre-allocate vertex arrays
                    VPos = np.zeros((nVertices, 3))
                    VColors = np.zeros((nVertices, 3))
                    ITris = np.zeros((nFaces, 3))
                if fields[0] == "COFF":
                    divideColor = True
            else:
                fields[0:3] = [int(field) for field in fields]
                [nVertices, nFaces, nEdges] = fields[0:3]
                VPos = np.zeros((nVertices, 3))
                VColors = np.zeros((nVertices, 3))
                ITris = np.zeros((nFaces, 3))
        elif vertex < nVertices:
            fields = [float(i) for i in fields]
            P = [fields[0],fields[1], fields[2]]
            color = np.array([0.5, 0.5, 0.5]) #Gray by default
            if len(fields) >= 6:
                #There is color information
                if divideColor:
                    color = [float(c)/255.0 for c in fields[3:6]]
                else:
                    color = [float(c) for c in fields[3:6]]
            VPos[vertex, :] = P
            VColors[vertex, :] = color
            vertex = vertex+1
        elif face < nFaces:
            #Assume the vertices are specified in CCW order
            fields = [int(i) for i in fields]
            ITris[face, :] = fields[1:fields[0]+1]
            face = face+1
    fin.close()
    VPos = np.array(VPos, np.float64)
    VColors = np.array(VColors, np.float64)
    ITris = np.array(ITris, np.int32)
    return (VPos, VColors, ITris)

def save_off(filename, VPos, VColors, ITris):
    """
    Save a .off file
    Parameters
    ----------
    filename: string
        Path to which to write .off file
    VPos : ndarray (N, 3)
        Array of points in 3D
    VColors : ndarray(N, 3)
        Array of RGB colors
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    """
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    fout = open(filename, "w")
    if VColors.size == 0:
        fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    else:
        fout.write("COFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g"%tuple(VPos[i, :]))
        if VColors.size > 0:
            fout.write(" %g %g %g"%tuple(VColors[i, :]))
        fout.write("\n")
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()
    
def get_face_centroids(VPos, ITris):
    """
    Return the centroids of each triangle face

    Parameters
    ----------
    VPos : ndarray (N, 3, dtype=float)
        Array of points in 3D
    ITris : ndarray (M, 3, dtype=int)
        Array of triangles connecting points, pointing to vertex indices
    
    Returns
    -------
    ndarray(M, 3, dtype=float)
        Face centroids
    """
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]
    return (P0 + P1 + P2)/3

class Vertex:
    def __init__(self, i, pos):
        self.i = i
        self.pos = pos
        self.neighbors = []
        self.touched = False

def dfs_cycle(v):
    v.touched = True
    cycle = [v.i]
    n = [c for c in v.neighbors if not c.touched]
    while len(n) > 0:
        v = n[0]
        v.touched = True
        cycle.append(v.i)
        n = [c for c in v.neighbors if not c.touched]
    return cycle

def get_cycles(cycle_edges, VPos):
    """
    Extract a list of cycles from the cycle edges

    Parameters
    ----------
    cycle_edges: ndarray(M, 2, dtype=int)
        List of edges whose union forms a bunch of cycles
    VPos: ndarray(K, 3)
        Positions of vertices that the cycles index into
    
    Returns
    -------
    list of N<M lists of int
        List of cycles, each of which is a list of indices
        into the vertex array
    vertices: list of N<M Vertex objects
        Vertex objects that the cycles index into
    """
    vertices = [Vertex(i, VPos[i, :]) for i in np.arange(np.max(cycle_edges)+1)]
    for [i, j] in cycle_edges:
        vertices[i].neighbors.append(vertices[j])
        vertices[j].neighbors.append(vertices[i])
    cycles = []
    for v in vertices:
        if not v.touched:
            cycle = dfs_cycle(v)
            cycles.append(cycle)
    return cycles, vertices

def get_spanning_bridges(cycles, cycle_edges, bridge_edges):
    """
    Return a subset of the bridge edges that spans the cycles

    Parameters
    ----------
    cycles: list of N<M lists of int
        List of cycles, each of which is a list of indices
        into the vertex array
    cycle_edges: ndarray(M, 2, dtype=int)
        List of edges whose union forms a bunch of cycles
    bridge_edges: ndarray(K, 2, dtype=int)
        List of edges that can form bridges
    
    Returns
    -------
    ndarray(L <= K, 2, dtype=int)
        Subset of bridge edges that span the cycles
    """
    from unionfind import UnionFind
    ## Step 1: Reindex bridge edges to have indices of unique cycles
    vertex2cycle = {i:-1 for i in np.unique(cycle_edges)}
    for idx, cycle in enumerate(cycles):
        for i in cycle:
            vertex2cycle[i] = idx
    
    ## Step 2: Construct a spanning tree between the cycles using
    ## the bridges
    N = len(cycles)
    uf = UnionFind(N)
    tree = [] # List of bridges to keep
    for idx, [i, j] in enumerate(bridge_edges):
        i = vertex2cycle[i]
        j = vertex2cycle[j]
        if not uf.find(i, j):
            uf.union(i, j)
            tree.append(idx)
    return np.array([bridge_edges[idx] for idx in tree], dtype=int)


def split_bridges(vertices, spanning_bridge_edges):
    """
    
    Parameters
    ----------
    vertices: list of N<M Vertex objects
        Vertex objects that the cycles index into
    ndarray(L, 2, dtype=int)
        Subset of bridge edges that span the cycles
    """
    idx = len(vertices) # Index of next new vertex to create
    dist = lambda x1, x2: np.sqrt(np.sum((x1.pos-x2.pos)**2))
    for [u, v] in spanning_bridge_edges:
        u = vertices[u]
        v = vertices[v]
        a, b = u.neighbors
        c, d = v.neighbors
        ## Step 1: Create split vertices
        w = Vertex(idx, 0.5*(u.pos+a.pos))
        vertices.append(w)
        idx += 1
        x = Vertex(idx, 0.5*(u.pos+b.pos))
        vertices.append(x)
        idx += 1
        y = Vertex(idx, 0.5*(v.pos+c.pos))
        vertices.append(y)
        idx += 1
        z = Vertex(idx, 0.5*(v.pos+d.pos))
        vertices.append(z)
        idx += 1
        ## Step 2: Figure out shorter assignment of bridges
        if dist(w, y) + dist(x, z) > dist(w, z) + dist(y, x):
            a, b = b, a
            w, x = x, w
        ## Step 3: Re-assign neighbors
        u.neighbors = [w, x]
        v.neighbors = [y, z]
        a.neighbors.remove(u)
        a.neighbors.append(w)
        b.neighbors.remove(u)
        b.neighbors.append(x)
        c.neighbors.remove(v)
        c.neighbors.append(y)
        d.neighbors.remove(v)
        d.neighbors.append(z)
        w.neighbors = [a, y]
        x.neighbors = [b, z]
        y.neighbors = [c, w]
        z.neighbors = [d, x]

def get_hamiltonian_cycle(VPos, ITris, prefix="", blossom_path="./blossom5"):
    """
    Implement the technique in [1]

    [1] Gopi, M., and David Eppstien. "Single‐strip triangulation 
    of manifolds with arbitrary topology." Computer Graphics Forum.
    Vol. 23. No. 3. Oxford, UK and Boston, USA: Blackwell Publishing,
    Inc, 2004.

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    prefix: string
        Prefix for out file for blossom, to avoid file collisions in the
        wrapped executable
    blossom_path: string
        Path to the blossom5 executable
    
    Returns
    -------
    """
    ## Step 1: Construct a dictionary from edges to triangle indices
    e2tris = {}
    for i in range(ITris.shape[0]):
        for k in range(3):
            i1, i2 = ITris[i, k], ITris[i, (k+1)%3]
            edge = (min(i1, i2), max(i1, i2))
            if not edge in e2tris:
                e2tris[edge] = set([])
            e2tris[edge].add(i)
    e2tris = {key:tuple(value) for key, value in e2tris.items()}
            
    ## Step 2: Construct dual graph based on edges by wrapping
    ## around blossom-V
    import subprocess
    import os
    fout = open("edges.txt", "w")
    fout.write("{} {}\n".format(ITris.shape[0], len(e2tris)))
    for edge in e2tris.values():
        edge = sorted(edge)
        fout.write("{} {} 1\n".format(*edge))
    fout.close()
    filename = "{}_blossom_out.txt".format(prefix)
    subprocess.call([blossom_path] + "-e edges.txt -w".split() + [filename])

    ## Step 3: Get a spanning subset of the bridge edges, split them,
    ## and construct the cycle
    FPos = get_face_centroids(VPos, ITris)
    bridge_edges = np.loadtxt(filename)[1::].astype(int)
    os.remove(filename)
    orig_edges = set([tuple([min(i, j), max(i, j)]) for [i, j] in e2tris.values()])
    cycle_edges = orig_edges.difference(set([tuple([min(i, j), max(i, j)]) for [i, j] in bridge_edges]))
    bridge_edges = np.array([x for x in bridge_edges], dtype=int)
    cycle_edges = np.array([x for x in cycle_edges], dtype=int)
    cycles, vertices = get_cycles(cycle_edges, FPos)
    spanning_bridge_edges = get_spanning_bridges(cycles, cycle_edges, bridge_edges)
    split_bridges(vertices, spanning_bridge_edges)
    X = np.array([v.pos for v in vertices])
    for v in vertices:
        v.touched = False
    cycle = dfs_cycle(vertices[-1])
    return X[cycle, :]