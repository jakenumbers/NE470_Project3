"""
solver.py
=========

Numerical solver for the multigroup finite-difference diffusion eigenvalue
problem.  Given the loss matrix ``A`` and fission matrix ``F`` produced by
:mod:`matrix_builder`, find ``(k, phi)`` such that::

        A . phi = (1/k) F . phi

The classical *power iteration* with sparse LU is used because A is a
banded sparse matrix (block-tridiagonal in space within each group, plus
sparse coupling between groups via in-scattering).  The reflection of the
unknowns is group-major (``index = g*N + i``).

Functions
---------
solve_keff(A, F, ...)
    Power iteration; returns (k, phi).

solve_adjoint(A, F, ...)
    Power iteration on (A^T, F^T); returns (k, phi*).

normalize_to_power_density(phi, mesh, target_W_per_cm3)
    Multiplicative rescaling of phi so that the fuel-volume-averaged
    power density matches ``target_W_per_cm3`` (default 47 W/cm^3,
    a representative PWR fuel value).  This is independent of any
    (artificial) cross-sectional area.

reshape_flux(phi, G, N)
    Rearranges the flat solution vector into ``phi[g, i]`` of shape (G, N).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
def reshape_flux(phi, G, N):
    """Convert flat group-major vector into shape (G, N)."""
    return phi.reshape(G, N)


def flatten_flux(phi_gx):
    """Convert (G, N) into flat group-major vector."""
    return np.asarray(phi_gx).reshape(-1)


# ---------------------------------------------------------------------------
def solve_keff_arpack(A, F):
    """Solve A phi = (1/k) F phi using ARPACK (scipy.sparse.linalg.eigs).

    This is typically much faster than power iteration when the dominance
    ratio is close to one (diffuse, large reactor with reflectors).

    Returns the same dict shape as :func:`solve_keff`.
    """
    A = sp.csc_matrix(A)
    F = sp.csc_matrix(F)

    # Largest k <-> largest eigenvalue of T = A^-1 F.  Use the shift-invert
    # trick by passing OPinv = inv(A) implicitly through eigs(F, M=A).
    vals, vecs = spla.eigs(F, k=1, M=A, which="LM", maxiter=5000, tol=1e-10)
    k = float(np.real(vals[0]))
    phi = np.real(vecs[:, 0])
    if phi.sum() < 0:
        phi = -phi
    return {
        "k": k,
        "phi": phi,
        "iters": None,
        "history": [k],
        "converged": True,
    }


def solve_keff(
    A,
    F,
    tol_k=1e-8,
    tol_phi=1e-7,
    max_iter=2000,
    verbose=False,
    phi_init=None,
    k_init=1.0,
    method="power",
):
    """
    Power iteration for ``A phi = (1/k) F phi``.

    Returns
    -------
    result : dict with keys
        k        : converged eigenvalue
        phi      : eigenvector (flat, group-major)
        iters    : number of outer iterations
        history  : list of k values per iteration
        converged: bool
    """
    if method == "arpack":
        return solve_keff_arpack(A, F)
    A = sp.csc_matrix(A)
    F = sp.csc_matrix(F)
    n = A.shape[0]

    if phi_init is None:
        phi = np.ones(n)
    else:
        phi = np.asarray(phi_init, dtype=float).copy()

    # Pre-factor A once for speed
    solve = spla.factorized(A)

    k = float(k_init)
    fission_old = F @ phi
    history = [k]

    converged = False
    it_used = 0
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

        # New eigenvalue estimate: ratio of integrated fission sources
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
        it_used = it

        if verbose and (it < 10 or it % 50 == 0):
            print(f"  iter {it:4d}  k = {k:.8f}  dk = {dk:.2e}  dphi = {diff:.2e}")

        if dk < tol_k and diff < tol_phi:
            converged = True
            break

    # Final positive-flux normalisation: max(phi) = 1 (by group)
    if phi.min() < 0 and abs(phi.min()) > abs(phi.max()):
        phi = -phi

    return {
        "k": float(k),
        "phi": phi,
        "iters": it_used,
        "history": history,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
def solve_adjoint(A, F, **kwargs):
    """Solve the adjoint problem A^T phi* = (1/k) F^T phi*.

    Returns the same dict structure as :func:`solve_keff` but the flux is
    the *adjoint* (importance) function and ``k`` should match the forward
    eigenvalue to within tolerance.
    """
    A_T = sp.csr_matrix(A).T.tocsr()
    F_T = sp.csr_matrix(F).T.tocsr()
    return solve_keff(A_T, F_T, **kwargs)


# ---------------------------------------------------------------------------
_KAPPA_DEFAULT = 3.2044e-11   # J/fission (~200 MeV) used as fallback
_NU_DEFAULT    = 2.43         # used only if kappa_sigma_f is missing


def _node_kappa_sigma_f(mat, g):
    """Return kappa Sigma_f for material ``mat`` group ``g``.

    Falls back to an estimate based on nu_sigma_f if kappa_sigma_f is not
    stored on the material (so a textbook material that only has
    nu_sigma_f still produces a meaningful power normalisation)."""
    if mat is None:
        return 0.0
    if mat.kappa_sigma_f is not None:
        return float(mat.kappa_sigma_f[g])
    if mat.sigma_f is not None:
        return float(mat.sigma_f[g]) * _KAPPA_DEFAULT
    if mat.nu_sigma_f is not None and mat.nu_sigma_f[g] > 0:
        return float(mat.nu_sigma_f[g]) / _NU_DEFAULT * _KAPPA_DEFAULT
    return 0.0


def _power_and_fuel_length(phi, mesh):
    """Return (integrated kappa Sigma_f * phi, fuel length).

    The first term is the integrated power per unit cross-sectional
    area [W/cm^2]; the second is the summed length of fuel-bearing
    half-cells [cm].  A half-cell is counted as fuel if its material
    has any group with nonzero kappa Sigma_f (or its sigma_f /
    nu_sigma_f fallbacks).  Both quantities use the same half-cell
    weighting the matrix builder employs.
    """
    G = mesh.G
    N = mesh.N
    phi_gx = reshape_flux(phi, G, N)
    p_per_area = 0.0
    L_fuel = 0.0
    sides = ((mesh.mat_left,  mesh.dx_left),
             (mesh.mat_right, mesh.dx_right))
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
                kf = _node_kappa_sigma_f(mat, g)
                p_per_area += kf * phi_gx[g, i] * dx_half
                if kf > 0:
                    has_fuel = True
            if has_fuel:
                L_fuel += dx_half
    return p_per_area, L_fuel


def average_power_density(phi, mesh):
    """
    Volume-averaged thermal power density [W/cm^3] over the
    fuel-bearing half-cells of the mesh.  In a 1-D slab this is
    independent of any (artificial) cross-sectional area::

        p_avg = (sum_{g,i} kappa Sigma_f * dx/2 * phi)_{fuel} / L_fuel
    """
    p_per_area, L = _power_and_fuel_length(phi, mesh)
    if L <= 0:
        raise RuntimeError("No fuel cells in mesh; cannot compute power density.")
    return p_per_area / L


def normalize_to_power_density(phi, mesh, target_W_per_cm3=47.0):
    """
    Rescale ``phi`` so that the fuel-averaged power density equals
    ``target_W_per_cm3`` (47 W/cm^3 by default, a representative PWR
    fuel power density).

    Returns the rescaled flux and the scaling factor.
    """
    p_now = average_power_density(phi, mesh)
    if p_now <= 0:
        raise RuntimeError("Cannot normalize: average power density is non-positive.")
    factor = target_W_per_cm3 / p_now
    return phi * factor, factor


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from materials import PWR_2G, H2O_2G
    from geometry import Region, Regions, Mesh
    from matrix_builder import build_matrices

    regs = Regions(
        Region("LeftRefl",  w=30.0,  n=31,  mat=H2O_2G),
        Region("Core",      w=300.0, n=151, mat=PWR_2G),
        Region("RightRefl", w=30.0,  n=31,  mat=H2O_2G),
    )
    m = Mesh(regs)
    A, F = build_matrices(m)
    res = solve_keff(A, F, verbose=False, tol_k=1e-9, tol_phi=1e-8,
                     method="arpack")
    print(f"\n  Forward (ARPACK)  k = {res['k']:.8f}")

    res_pi = solve_keff(A, F, verbose=False, tol_k=1e-9, tol_phi=1e-8,
                        method="power", max_iter=5000)
    print(f"  Forward (power)   k = {res_pi['k']:.8f}, "
          f"iters = {res_pi['iters']}, conv = {res_pi['converged']}")

    res_adj = solve_adjoint(A, F, verbose=False, method="arpack")
    print(f"  Adjoint  (ARPACK) k = {res_adj['k']:.8f}")

    # Check power-density normalisation
    phi_norm, _ = normalize_to_power_density(res["phi"], m,
                                             target_W_per_cm3=47.0)
    p_avg = average_power_density(phi_norm, m)
    print(f"  Normalized avg power density = {p_avg:.3f} W/cm^3 (target 47.0)")
