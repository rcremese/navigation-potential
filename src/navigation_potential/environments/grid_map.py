import numpy as np

class GridMap:
    def __init__(self, dim_sizes : tuple[int], leaf_size : float = 1.0):
        self.dim_sizes = np.array(dim_sizes, dtype=np.uint32)
        self.ndims = len(self.dim_sizes)
        self.leaf_size = leaf_size
        self.ncells = np.prod(self.dim_sizes)
        self.clean = True

        # Initialize arrays
        self.cells = np.zeros(self.ncells, dtype=object)
        self.d = np.cumprod(np.concatenate(([1], self.dim_sizes[:-1])))
        self.occupied = []

        # Set cell indices
        coords = np.unravel_index(np.arange(self.ncells), self.dim_sizes)
        for idx, coord in enumerate(np.transpose(coords)):
            self.cells[idx] = Cell(idx, coord)

    def __getitem__(self, idx):
        return self.cells[idx]

    def get_cell(self, idx):
        return self.cells[idx]

    def get_min_value_in_dim(self, idx, dim):
        coords = self.idx2coord(idx)
        neighs = []
        for d in range(self.ndims):
            if d == dim:
                continue
            c1 = coords.copy()
            c2 = coords.copy()
            c1[d] = max(0, c1[d] - 1)
            c2[d] = min(self.dim_sizes[d] - 1, c2[d] + 1)
            neighs.append(self.coord2idx(c1))
            neighs.append(self.coord2idx(c2))

        values = [self.cells[n].value for n in neighs]
        return min(values)

    def get_neighbors(self, idx):
        coords = self.idx2coord(idx)
        neighs = []
        for d in range(self.ndims):
            c1 = coords.copy()
            c2 = coords.copy()
            c1[d] = max(0, c1[d] - 1)
            c2[d] = min(self.dim_sizes[d] - 1, c2[d] + 1)
            neighs.append(self.coord2idx(c1))
            neighs.append(self.coord2idx(c2))
        return neighs

    def idx2coord(self, idx):
        coords = np.zeros(self.ndims, dtype=np.uint32)
        for d in range(self.ndims - 1, -1, -1):
            coords[d] = idx // self.d[d]
            idx -= coords[d] * self.d[d]
        return coords

    def coord2idx(self, coords):
        idx = coords[0]
        for d in range(1, self.ndims):
            idx += coords[d] * self.d[d - 1]
        return idx

    def size(self):
        return self.ncells

    def get_max_value(self):
        values = [c.value for c in self.cells if not np.isinf(c.value)]
        return max(values) if values else 0.0

    def is_clean(self):
        return self.clean

    def set_clean(self, clean):
        self.clean = clean

    def clean_grid(self):
        if not self.clean:
            for c in self.cells:
                c.set_default()
            self.clean = True

    def clear(self):
        self.cells = np.zeros(self.ncells, dtype=object)
        self.occupied = []

    def get_dim_sizes_str(self):
        return ' '.join(str(d) for d in self.dim_sizes)

    def set_occupied_cells(self, occupied):
        self.occupied = occupied

    def get_occupied_cells(self):
        return self.occupied

    def get_ndims(self):
        return self.ndims

    def get_avg_speed(self):
        speeds = [c.velocity for c in self.cells if not c.is_occupied]
        return np.mean(speeds) if speeds else 0.0

    def get_max_speed(self):
        speeds = [c.velocity for c in self.cells]
        return max(speeds) if speeds else 0.0

class Cell:
    def __init__(self, idx, coord):
        self.idx = idx
        self.coord = coord
        self.value = float('inf')
        self.velocity = 1.0
        self.occupied = False

    def set_default(self):
        self.value = float('inf')
        self.velocity = 1.0
        self.occupied = False

    def is_occupied(self):
        return self.occupied

    def get_velocity(self):
        return self.velocity
