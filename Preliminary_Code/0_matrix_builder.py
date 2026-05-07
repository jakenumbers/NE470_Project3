"""
matrix_builder.py
=================

Build the loss (A) and fission-production (F) matrices for the multigroup
finite-difference diffusion eigenvalue problem::

        A . phi = (1/k) F . phi

The construction follows the *interface-node* method shown in the project
handout.  For an interior node fully inside material ``m`` with uniform
spacing ``dx`` the integrated equation is::

   - (D_m/dx) phi_{i-1}
   + (2 D_m/dx + Sigma_R,m * dx) phi_i
   - (D_m/dx) phi_{i+1}
   - sum_{g'!=g} Sigma_s,g'->g,m * dx * phi_{g',i}
   = (1/k) * chi_g * sum_{g'} nu Sigma_f,g',m * dx * phi_{g',i}

For an interface node between materials A (left) and B (right) with
spacings dx_A and dx_B, the same integrated equation specialises to::

   - (D_A/dx_A) phi_{i-1}
   + ((D_A/dx_A) + (D_B/dx_B)
        + Sigma_R,g,A * dx_A/2 + Sigma_R,g,B * dx_B/2) phi_i
   - (D_B/dx_B) phi_{i+1}
   - sum_{g'!=g} (Sigma_s,g'->g,A * dx_A/2 + Sigma_s,g'->g,B * dx_B/2) phi_{g',i}
   = (1/k) * sum_{g'} (chi_g^A * nu Sigma_f,g',A * dx_A/2
                        + chi_g^B * nu Sigma_f,g',B * dx_B/2) * phi_{g',i}

The internal-node form is recovered exactly when material A == material B
and dx_A == dx_B.

The interior layout used here is **group-major**::

    linear index = g * N + i,  i in [0, N-1], g in [0, G-1]

so the full matrix is a (G x G) block of (N x N) tridiagonal blocks.

Boundary conditions: at the global left and right edges (nodes 0 and N-1)
we apply a zero-flux Dirichlet BC by replacing the corresponding rows of
A with the identity and zeroing the corresponding rows of F.  The
solution then has phi(boundary) = 0 automatically.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from geometry import Mesh


def build_matrices(mesh: Mesh, return_dense: bool = False):
    """
    Build the loss matrix ``A`` and fission matrix ``F``.

    Parameters
    ----------
    mesh : Mesh
        Pre-built mesh (geometry + materials).
    return_dense : bool
        If True, dense numpy arrays are returned in addition to sparse
        CSR matrices.

    Returns
    -------
    A, F : scipy.sparse.csr_matrix
        Each of size (G*N, G*N).
    (optionally) A_dense, F_dense : np.ndarray
    """
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
            # Zero-flux Dirichlet: phi_{g,i} = 0 for every group
            for g in range(G):
                row = lin(g, i)
                A[row, row] = 1.0
                # F row stays all zeros => 0 = (1/k)*0 trivially satisfied
            continue

        dxL = mesh.dx_left[i]
        dxR = mesh.dx_right[i]
        matL = mesh.mat_left[i]
        matR = mesh.mat_right[i]

        # Diffusion couplings to neighbours - same for any group
        for g in range(G):
            DL = matL.D[g] / dxL if matL is not None else 0.0
            DR = matR.D[g] / dxR if matR is not None else 0.0

            row = lin(g, i)

            # Off-diagonal spatial neighbours (same group block):
            if i - 1 >= 0:
                A[row, lin(g, i - 1)] += -DL
            if i + 1 < N:
                A[row, lin(g, i + 1)] += -DR

            # Diagonal: diffusion + integrated removal (sigma_r)
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
            #  with each side weighted by its half-cell.
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

    if return_dense:
        return A, F, A.toarray(), F.toarray()
    return A, F


# ---------------------------------------------------------------------------
# Optional: a dedicated kappa-fission matrix used for power normalisation.
# ---------------------------------------------------------------------------
def build_kappa_matrix(mesh: Mesh):
    """
    Build a (G*N, G*N) sparse 'energy' matrix K such that the integrated
    thermal power per unit cross-sectional area is::

        P_per_area = phi^T . K . phi / phi_norm    (in J/cm^2/s = W/cm^2)

    when phi has units of n/cm^2/s.  K is diagonal in space (same node)
    and acts as ``kappa Sigma_f`` integrated over the half-cell volume of
    each side.  In a 1-D slab with cross-sectional area A_cross, the total
    power is::

        P [W] = A_cross * sum_{g,i} kappa Sigma_f,g(i) * phi_g(i) * v_i

    where ``v_i`` is the half-cell volume around node i.  The K-matrix
    convention is::

        K[g*N+i, g*N+i] = kappa Sigma_f,g^L * dx_L/2 + kappa Sigma_f,g^R * dx_R/2
    """
    G = mesh.G
    N = mesh.N
    size = G * N
    K = sp.lil_matrix((size, size), dtype=float)
    for i in range(N):
        if mesh.is_boundary[i]:
            continue
        for g in range(G):
            v = 0.0
            if mesh.mat_left[i] is not None and mesh.mat_left[i].kappa_sigma_f is not None:
                v += mesh.mat_left[i].kappa_sigma_f[g] * mesh.dx_left[i] * 0.5
            if mesh.mat_right[i] is not None and mesh.mat_right[i].kappa_sigma_f is not None:
                v += mesh.mat_right[i].kappa_sigma_f[g] * mesh.dx_right[i] * 0.5
            row = g * N + i
            K[row, row] = v
    return K.tocsr()


if __name__ == "__main__":
    from materials import PWR_2G, H2O_2G
    from geometry import Region, Regions, Mesh

    regs = Regions(
        Region("LeftRefl",  w=20.0,  n=21,  mat=H2O_2G),
        Region("Core",      w=200.0, n=101, mat=PWR_2G),
        Region("RightRefl", w=20.0,  n=21,  mat=H2O_2G),
    )
    msh = Mesh(regs)
    A, F = build_matrices(msh)
    print("A shape =", A.shape, " nnz =", A.nnz)
    print("F shape =", F.shape, " nnz =", F.nnz)
    # Sanity: A must be invertible (nonzero rows for boundary BCs)
    print("A diagonal min =", A.diagonal().min(), " max =", A.diagonal().max())
