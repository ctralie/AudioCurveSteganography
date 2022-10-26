class UnionFind:
    def __init__(self, N):
        self.parent = [i for i in range(N)]
        self.size = [1 for i in range(N)]
        self._operations = 0
        self._calls = 0
    
    def root(self, i):
        if i != self.parent[i]:
            self.parent[i] = self.root(self.parent[i])
            self._operations += 1
        return self.parent[i]
    
    def find(self, i, j):
        self._calls += 1
        return self.root(i) == self.root(j)
    
    def union(self, i, j):
        self._calls += 1
        rooti = self.root(i)
        rootj = self.root(j)
        if rooti != rootj:
            if self.size[rooti] < self.size[rootj]:
                self.parent[rooti] = rootj
                self.size[rootj] += self.size[rooti]
            else:
                self.parent[rootj] = rooti
                self.size[rooti] += self.size[rootj]
            self._operations += 1
