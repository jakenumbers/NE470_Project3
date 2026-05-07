"""
matrix_builder.py
-----------------
Builds matrix using the mesh defined in the geometry.py
Uses scipy's sparse matrix tools and slowly builds the matrix node by node

"""

import scipy.sparse as sp
from geometry import Mesh

#%% Matrix Builder using generated mesh
def build_matrices(mesh: Mesh):
    """
    Build the matrix A and matrix F.

    """
    #Pulling in values from mesh  and determines array sizes
    G = mesh.G
    N = mesh.N
    size = G * N

    # Use lil_matrix for incremental construction, convert at the end.
    A = sp.lil_matrix((size, size), dtype=float)
    F = sp.lil_matrix((size, size), dtype=float)

    def lin(g, i):
        return g * N + i

    for i in range(N):
        if mesh.is_boundary[i]:
            # Zero-flux: phi_{g,i} = 0 for every group
            for g in range(G):
                row = lin(g, i)
                A[row, row] = 1.0
                # F row stays all zeros => 0 = (1/k)*0 trivially satisfied
            continue

        dxL = mesh.dx_l[i]
        dxR = mesh.dx_r[i]
        matL = mesh.mat_l[i]
        matR = mesh.mat_r[i]

        # Diffusion couplings to neighbours - same for any group (DL = DR if Material is the same)
        for g in range(G):
            DL = matL.D[g] / dxL if matL is not None else 0.0
            DR = matR.D[g] / dxR if matR is not None else 0.0

            row = lin(g, i)

            # Off-diagonals (same group block):
            if i - 1 >= 0:
                A[row, lin(g, i - 1)] += -DL
            if i + 1 < N:
                A[row, lin(g, i + 1)] += -DR

            # Main Diagonal: Diffusion + Removal (sigma_r) (If interior node DL and DR are the same)
            sR_int = 0.0
            if matL is not None:
                sR_int += matL.sigma_r[g] * dxL * 0.5
            if matR is not None:
                sR_int += matR.sigma_r[g] * dxR * 0.5
            A[row, row] += DL + DR + sR_int

            # In-scattering from other groups g' -> g (off-diagonal blocks of A)
            for gp in range(G):
                if gp == g:
                    continue
                ss_int = 0.0
                if matL is not None:
                    ss_int += matL.sigma_s[gp, g] * dxL * 0.5
                if matR is not None:
                    ss_int += matR.sigma_s[gp, g] * dxR * 0.5
                if ss_int != 0.0:
                    A[row, lin(gp, i)] += -ss_int

            # Fission production: F[g·N+i, g'·N+i] = chi_g * nu Sigma_f,g'
            #  with each side weighted by its half-node.
            for gp in range(G):
                f_int = 0.0
                if matL is not None:
                    f_int += (matL.chi[g] * matL.nu_sigma_f[gp]
                              * dxL * 0.5)
                if matR is not None:
                    f_int += (matR.chi[g] * matR.nu_sigma_f[gp]
                              * dxR * 0.5)
                if f_int != 0.0:
                    F[row, lin(gp, i)] += f_int

    A = A.tocsr()
    F = F.tocsr()

    return A, F

