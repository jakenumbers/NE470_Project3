"""
geometry.py
===========

1-D layered slab geometry plus the material-attached mesh that the
matrix builder consumes.

A :class:`Region` has:

    name : human-readable label
    w    : width of the region                                 [cm]
    n    : number of *nodes* contributed by this region
    mat  : Material

Regions are laid end-to-end and the interface node between adjacent
regions is shared, so the total number of nodes in the mesh is::

    N_total = sum(n_r) - (num_regions - 1)

Each region uses uniform spacing ``dx_r = w_r / (n_r - 1)``.

:class:`Regions` lays out the cells and node positions:

    x[i]            : node positions                           [cm]
    cell_dx[k]      : spacing of cell k (k = 0 .. N-2)         [cm]
    cell_region[k]  : index of the region cell k belongs to

:class:`Mesh` wraps a :class:`Regions` and exposes per-node
"left half-cell" / "right half-cell" arrays the matrix builder uses:

    dx_left[i]      : spacing of cell to the left of node i  (0 at left bdy)
    dx_right[i]     : spacing of cell to the right of node i (0 at right bdy)
    mat_left[i]     : material to the left of node i  (None at left bdy)
    mat_right[i]    : material to the right of node i (None at right bdy)
    is_boundary[i]  : True at the global edge nodes

For an interface node between materials A (left) and B (right),
``mat_left = A`` with ``dx_A`` and ``mat_right = B`` with ``dx_B``,
so the matrix builder gets ``dx_A/2`` of A on the left and ``dx_B/2``
of B on the right.

Boundary conditions are not encoded here -- that is handled in the
matrix builder.
"""

from __future__ import annotations

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


class Regions:
    """Ordered collection of :class:`Region` objects -> a 1-D mesh layout."""

    def __init__(self, *regions):
        if not regions:
            raise ValueError("At least one Region is required.")
        for r in regions:
            if not isinstance(r, Region):
                raise TypeError(f"Unsupported region descriptor: {r!r}")

        G = regions[0].mat.G
        for r in regions:
            if r.mat.G != G:
                raise ValueError(
                    f"All materials must share the same group count G={G}; "
                    f"region {r.name!r} has G={r.mat.G}."
                )

        self.regions = list(regions)
        self.G = G

        # Each region contributes (n_r - 1) cells.
        self.cell_dx = np.concatenate(
            [np.full(r.n - 1, r.dx) for r in self.regions]
        )
        self.cell_region = np.concatenate(
            [np.full(r.n - 1, idx, dtype=int)
             for idx, r in enumerate(self.regions)]
        )
        self.x = np.concatenate([[0.0], np.cumsum(self.cell_dx)])
        self.N = self.x.size
        self.total_width = float(self.x[-1])

    def cell_material(self, k):
        """Material of cell ``k`` (the cell between node k and k+1)."""
        return self.regions[self.cell_region[k]].mat

    def summary(self):
        lines = [
            f"Regions:  total width = {self.total_width:.4f} cm,  "
            f"{self.N} nodes,  G = {self.G}"
        ]
        for r in self.regions:
            lines.append(
                f"   {r.name:<20s} w={r.w:8.3f} cm  n={r.n:4d}  "
                f"dx={r.dx:8.4f} cm  mat={r.mat.name}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"Regions(N={self.N}, total_width={self.total_width:.3f} cm)"


class Mesh:
    """Material-attached mesh that the matrix builder consumes.

    Each interior node has half a cell of material on each side;
    interface nodes between materials A and B carry ``dx_A/2`` of A on
    the left and ``dx_B/2`` of B on the right.  The matrix builder
    combines these into the integrated finite-difference equations.
    """

    def __init__(self, regions: Regions):
        self.regions = regions
        self.G = regions.G
        self.N = regions.N
        self.x = regions.x.copy()

        # Cell k lies between nodes k and k+1, so cell (i-1) is on the
        # left of node i and cell i is on the right.  Pad with zeros at
        # the boundaries (where there is no cell on that side).
        self.dx_left = np.concatenate([[0.0], regions.cell_dx])
        self.dx_right = np.concatenate([regions.cell_dx, [0.0]])

        cells = [regions.cell_material(k) for k in range(regions.N - 1)]
        self.mat_left = [None, *cells]
        self.mat_right = [*cells, None]

        self.is_boundary = np.zeros(self.N, dtype=bool)
        self.is_boundary[0] = True
        self.is_boundary[-1] = True

    def summary(self):
        return self.regions.summary() + (
            f"\nMesh:  G = {self.G}, N = {self.N}, dx range = "
            f"[{self.regions.cell_dx.min():.4f}, "
            f"{self.regions.cell_dx.max():.4f}] cm"
        )


if __name__ == "__main__":
    from materials import PWR_2G, H2O_2G

    regs = Regions(
        Region("LeftRefl",  w=20.0,  n=11,  mat=H2O_2G),
        Region("Core",      w=200.0, n=101, mat=PWR_2G),
        Region("RightRefl", w=20.0,  n=11,  mat=H2O_2G),
    )
    mesh = Mesh(regs)
    print(mesh.summary())

    iface = 10        # left interface between H2O and PWR
    print(f"\nNode {iface}: x = {mesh.x[iface]:.3f}")
    print(f"  left  mat = {mesh.mat_left[iface].name},  dx_L = {mesh.dx_left[iface]}")
    print(f"  right mat = {mesh.mat_right[iface].name},  dx_R = {mesh.dx_right[iface]}")
