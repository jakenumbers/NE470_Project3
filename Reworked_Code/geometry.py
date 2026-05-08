"""
geometry.py
-----------
A Region has:

    name : region name
    w    : width of the region                                 [cm]
    n    : number of *nodes* contributed by this region
    mat  : Material

Regions are laid end-to-end and the interface node between adjacent
regions is shared, so the total number of nodes in the mesh is:
    
    N_total = sum(n_r) - (num_regions - 1)

Regions Class lays out the nodes and node positions:

    x[i]            : node positions                           [cm]
    node_dx[k]      : spacing of node k (k = 0 .. N-2)         [cm]
    node_region[k]  : index of the region node k belongs to

Mesh Class contains per-node materials (Used to determine if a node is interior or interface)
"left half-node" / "right half-node" arrays the matrix builder uses:

    dx_l[i]      : spacing of node to the left of node i  (0 at boundary)
    dx_r[i]     : spacing of node to the right of node i (0 at boundary)
    mat_l[i]     : material to the left of node i  (None at left boundary)
    mat_r[i]    : material to the right of node i (None at right boundary)
    is_boundary[i]  : only True at the global edge nodes

"""
import numpy as np

class Region:
    """One slab region with uniform material and uniform spacing."""

    def __init__(self, name, w, n, mat):
        if n < 2:
            raise ValueError(f"Region {name!r}: n must be >= 2 (got {n}).")
        if w <= 0:
            raise ValueError(f"Region {name!r}: w must be > 0 (got {w}).")

        self.name = str(name)
        self.w = float(w)
        self.n = int(n)
        self.mat = mat
        self.dx = self.w / (self.n - 1)

    def __repr__(self):
        return (f"Region(name={self.name!r}, w={self.w}, n={self.n}, "
                f"mat={self.mat.name!r})")

#%% Puts individual regions together into one array and sets node materials
class Regions:
    """Combines individual Regions together."""

    def __init__(self, *regions):
        if not regions:
            raise ValueError("At least one Region is required.")
        G = regions[0].mat.G
        #Check for all materials have the same group
        for r in regions:
            if r.mat.G != G:
                raise ValueError(
                    f"All materials must share the same group count G={G}; "
                    f"region {r.name!r} has G={r.mat.G}.")
                
        self.regions = list(regions)
        self.G = G

        # Each region contributes (n_r - 1) nodes.
        self.node_dx = np.concatenate(
            [np.full(r.n - 1, r.dx) for r in self.regions]
        )
        self.node_region = np.concatenate(
            [np.full(r.n - 1, idx, dtype=int)
             for idx, r in enumerate(self.regions)]
        )
        #Creates X-array advancing every dx for the different regions
        self.x = np.concatenate([[0.0], np.cumsum(self.node_dx)])
        #Defines total number of nodes in array
        self.N = self.x.size
        #Total width of region of all regions
        self.total_width = float(self.x[-1])

    def node_material(self, k):
        """Material of node i (the node between node k and k+1)."""
        return self.regions[self.node_region[k]].mat

    def summary(self):
        lines = [
            f"Regions: w_tot={self.total_width:.4f} cm, " 
            f"{self.N} nodes, G={self.G}"]

class Mesh:
    """Material-attached mesh that the matrix builder consumes.

    Each interior node has half a node of material on each side;
    interface nodes between materials A and B carry `dx_A/2` of A on
    the left and `dx_B/2` of B on the right.  The matrix builder
    combines these into the integrated finite-difference equations.
    """

    def __init__(self, regions: Regions):
        self.regions = regions
        self.G = regions.G
        self.N = regions.N
        self.x = regions.x.copy()

        # This sets the dx for left and right side of the nodes
        self.dx_l = np.concatenate([[0.0], regions.node_dx])
        self.dx_r = np.concatenate([regions.node_dx, [0.0]])
        
        #This sets the material for left and right side of the node (same material is an interior node)
        nodes = [regions.node_material(k) for k in range(regions.N - 1)]
        self.mat_l = [None, *nodes]
        self.mat_r = [*nodes, None]
        
        #Creates a boundary boolean array for the boundaries of the region
        self.is_boundary = np.zeros(self.N, dtype=bool)
        self.is_boundary[0] = True
        self.is_boundary[-1] = True
    
    #Summary function of mesh. Gives number of groups, nodes, minimum and maximum change in x
    def summary(self):
        return self.regions.summary() + (
            f"\nMesh:  G = {self.G}, N = {self.N}, dx range = "
            f"[{self.regions.node_dx.min():.4f}, "
            f"{self.regions.node_dx.max():.4f}] cm")

