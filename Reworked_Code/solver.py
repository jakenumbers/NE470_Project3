"""
solver.py
------------
Functions

solve_keff(A, F, ...)
    Power iteration; returns (k, phi).

solve_adjoint(A, F, ...)
    Power iteration on (A^T, F^T); returns (k, phi*).

normalize_to_power_density(phi, mesh, target_W_per_cm3)
    Multiplicative rescaling of phi so that the fuel-volume-averaged
    power density matches target_W_per_cm3.

reshape_flux(phi, G, N)
    Rearranges the flat solution vector into phi[g, i] of shape (G, N).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


#%% Flux Reshape for power normalization and plotting
def reshape_flux(phi, G, N):
    """Convert flat group-major vector into shape (G, N)."""
    return phi.reshape(G, N)

#%% Flatten back to solving flux
def flatten_flux(phi_gx):
    """Convert (G, N) into flat group-major vector."""
    return np.asarray(phi_gx).reshape(-1)

#%% Power Iteration after matrices are built
def solve_keff(
    A,
    F,
    tol_k=1e-8,
    tol_phi=1e-7,
    max_iter=1000,
    phi_init=None,
    k_init=1.0,):
    """
    Power iteration for `A phi = (1/k) F phi`.

    Returns
    -------
    result : dict with keys
        k        : converged eigenvalue
        phi      : eigenvector (flat, group-major)
        iters    : number of outer iterations
        history  : list of k values per iteration
        converged: bool
    """
    A = sp.csc_matrix(A)
    F = sp.csc_matrix(F)
    n = A.shape[0]

    if phi_init is None:
        phi = np.ones(n)
    else:
        phi = np.asarray(phi_init, dtype=float)

    # Pre-factor A once for speed
    solve = spla.factorized(A)
    #Input starting values of k, fission and begin tracking k
    k = float(k_init)
    fission_old = F @ phi
    history = [k]

    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        # Power iteration on T = A^-1 F.  Solving A phi_new = F phi_old
        # gives phi_new = T phi_old, which converges to k_eff * phi_eig.
        rhs = fission_old.copy()
        phi_new = solve(rhs)

        fission_new = F @ phi_new
        s_new = fission_new.sum()
        s_old = fission_old.sum()
        if s_old == 0:
            raise RuntimeError("Fission source vanished during iteration.")

        # New estimate: ratio of integrated fission sources
        k_new = s_new / s_old

        # Normalise phi so integrated fission stays ~1 (avoids overflow).
        if s_new != 0:
            phi_new = phi_new / s_new
            fission_new = fission_new / s_new

        # Convergence
        dk = abs(k_new - k) / max(abs(k_new), 1e-30)
        diff = np.linalg.norm(phi_new - phi) / max(np.linalg.norm(phi_new), 1e-30)

        phi = phi_new
        fission_old = fission_new
        k = k_new
        history.append(k)
        iterations = it

        if dk < tol_k and diff < tol_phi:
            converged = True
            break

    # Final positive-flux normalisation: max(phi) = 1 (by group)
    if phi.min() < 0 and abs(phi.min()) > abs(phi.max()):
        phi = -phi

    return {
        "k": float(k),
        "phi": phi,
        "iters": iterations,
        "history": history,
        "converged": converged,
    }


#%% Solving Transpose of A and F matrices 
def solve_adjoint(A, F, **kwargs):
    """Solve the adjoint problem A^T phi* = (1/k) F^T phi*.

    Returns the same dict structure as :func:solve_keff but the flux is
    the adjoint (importance) function and k should match the forward
    eigenvalue to within tolerance.
    """
    A_T = sp.csr_matrix(A).T.tocsr()
    F_T = sp.csr_matrix(F).T.tocsr()
    return solve_keff(A_T, F_T, **kwargs)

#%% Power Density Normalization
Kf = 3.2044e-11   # J/fission (~200 MeV)

def node_sigma_f(mat, g):

    if mat is None:
        return 0.0
    if mat.sigma_f is not None:
        return float(mat.sigma_f[g])

def power_and_fuel_width(phi, mesh):
    """Return (Kf * sigma_f * phi, fuel width).

    The first term is the integrated power per unit cross-sectional
    area [W/cm^2]; the second is the summed length of fuel-bearing
    nodes [cm].  A node is counted as fuel if its material
    has any group with nonzero kf_sigma_f.
    """
    G = mesh.G
    N = mesh.N
    phi_gx = reshape_flux(phi, G, N)
    p_per_area = 0.0
    W_fuel = 0.0
    sides = ((mesh.mat_l,  mesh.dx_l),
             (mesh.mat_r, mesh.dx_r))
    for i in range(N):
        if mesh.is_boundary[i]:
            continue
        for mat_arr, dx_arr in sides:
            mat = mat_arr[i]
            if mat is None:
                continue
            dx_half = dx_arr[i] * 0.5
            has_fuel = False
            for g in range(G):
                kf_sig_f = node_sigma_f(mat, g) * Kf
                p_per_area += kf_sig_f * phi_gx[g, i] * dx_half
                if kf_sig_f > 0:
                    has_fuel = True
            if has_fuel:
                W_fuel += dx_half
    return p_per_area, W_fuel


def average_power_density(phi, mesh):
    """
    Volume-averaged thermal power density [W/cm^3] over thefuel-bearing nodes of the mesh.
    """
    p_per_area, W = power_and_fuel_width(phi, mesh)
    if W <= 0:
        raise RuntimeError("No fuel cells in mesh; cannot compute power density.")
    return p_per_area / W


def normalize_to_power_density(phi, mesh, target_W_per_cm3=45):
    Power = 300e6 # Watts
    """
    Rescale phi so that the fuel-averaged power density equals target_W_per_cm3

    Returns the rescaled flux and the scaling factor.
    """
    p_now = average_power_density(phi, mesh)
    # l_w = (mesh.regions.total_width*25) # length and width dimensions, 25x thickness to assume infinite slab
    
    # factor = (Power/(l_w**2*mesh.regions.total_width))/p_now
    # print(f'Depth/Height of Slab: {l_w} cm')
    # print(f'Thickness of Slab: {mesh.regions.total_width} cm')

    factor = target_W_per_cm3 / p_now
    return phi * factor, factor

