"""
perturbation.py
===============

Adjoint solution and Rayleigh-quotient perturbation theory for the
extra-credit portion of the project.

Theory
------
The forward eigenvalue problem is::

    A . phi = (1/k) F . phi

with adjoint::

    A^T . phi* = (1/k) F^T . phi*

For any forward-adjoint pair the eigenvalue is given exactly by the
Rayleigh quotient::

    k = <phi*, F phi> / <phi*, A phi>                          (1)

For *small* perturbations dA, dF (with respect to A, F) the first-order
prediction of the eigenvalue change is::

    dk ~= ( <phi*, dF phi> - k * <phi*, dA phi> ) / <phi*, A phi>   (2)

This module provides

    rayleigh_k(phi, phi_adj, A, F)
        Evaluate (1).

    perturb_absorption(mesh, mat_name, group, dsigma_a)
        Build a *perturbation* (dA, dF=0) that corresponds to changing
        the absorption cross section of one material/group by
        ``dsigma_a``.  The change propagates into ``sigma_r`` so the
        diagonal of A picks up the correct contribution; F is unchanged.

    predict_dk_first_order(phi, phi_adj, A, F, dA, dF, k)
        Apply (2).

    sweep_absorption(...)
        Convenience: vary ``dsigma_a`` over a range, re-solving the
        perturbed problem each time, and compare the exact dk against
        the first-order prediction.  Returns arrays suitable for
        plotting in :mod:`post_processing`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp

from materials import Material
from geometry import Region, Regions, Mesh
from matrix_builder import build_matrices
from solver import solve_keff, solve_adjoint


# ---------------------------------------------------------------------------
def rayleigh_k(phi, phi_adj, A, F):
    """Return the Rayleigh-quotient eigenvalue estimate."""
    Aphi = A @ phi
    Fphi = F @ phi
    num = float(np.dot(phi_adj, Fphi))
    den = float(np.dot(phi_adj, Aphi))
    if den == 0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
def predict_dk_first_order(phi, phi_adj, A, F, dA, dF, k):
    """First-order perturbation prediction of dk from (dA, dF)."""
    Aphi = A @ phi
    dAphi = dA @ phi
    dFphi = dF @ phi
    num = float(np.dot(phi_adj, dFphi)) - k * float(np.dot(phi_adj, dAphi))
    den = float(np.dot(phi_adj, Aphi))
    if den == 0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
def _perturbed_material(mat: Material, *, dsigma_a=None, dnu_sigma_f=None,
                        group: Optional[int] = None):
    """
    Build a copy of ``mat`` with absorption (and/or nu*Sigma_f) shifted.

    If ``group`` is None, the perturbation is applied to all groups; else
    only the specified group (0-based).
    """
    new = copy.deepcopy(mat)

    if dsigma_a is not None:
        delta = np.zeros_like(new.sigma_a)
        if group is None:
            delta[:] = dsigma_a
        else:
            delta[group] = dsigma_a
        new.sigma_a = new.sigma_a + delta
        # Removal includes absorption + scatter-out; absorption-only
        # perturbations propagate directly to sigma_r.
        new.sigma_r = new.sigma_r + delta

    if dnu_sigma_f is not None:
        delta_f = np.zeros_like(new.nu_sigma_f)
        if group is None:
            delta_f[:] = dnu_sigma_f
        else:
            delta_f[group] = dnu_sigma_f
        new.nu_sigma_f = new.nu_sigma_f + delta_f

    return new


def _regions_with_material_replaced(regions: Regions, target_mat: Material,
                                    new_mat: Material) -> Regions:
    """Return a new Regions where any region using ``target_mat`` is
    replaced with one using ``new_mat`` (same width / node count)."""
    new_regions = []
    for r in regions.regions:
        if r.mat is target_mat:
            new_regions.append(Region(name=r.name, w=r.w, n=r.n, mat=new_mat))
        else:
            new_regions.append(r)
    return Regions(*new_regions)


# ---------------------------------------------------------------------------
@dataclass
class PerturbationResult:
    dsigma_a: float
    k_perturbed: float
    dk_exact: float
    dk_predicted: float


def sweep_absorption(
    base_regions: Regions,
    target_material: Material,
    group: int,
    dsigma_a_values: Iterable[float],
    method: str = "arpack",
    verbose: bool = True,
):
    """
    For a sequence of absorption-perturbation magnitudes ``dsigma_a_values``
    (added to ``target_material`` in ``group``), solve the perturbed
    eigenvalue problem and compare the first-order Rayleigh-quotient
    prediction to the exact dk.

    Returns
    -------
    base : dict with k0, phi, phi_adj, A, F
    results : list[PerturbationResult]
    """
    # Reference forward + adjoint solve
    mesh = Mesh(base_regions)
    A, F = build_matrices(mesh)
    fwd = solve_keff(A, F, method=method)
    adj = solve_adjoint(A, F, method=method)
    k0 = fwd["k"]
    phi = fwd["phi"]
    phi_adj = adj["phi"]

    # Verify Rayleigh quotient on the unperturbed problem
    k_RQ = rayleigh_k(phi, phi_adj, A, F)
    if verbose:
        print(f"  Reference k_eff (forward) = {k0:.6f}")
        print(f"  Adjoint k_eff             = {adj['k']:.6f}")
        print(f"  Rayleigh-quotient k       = {k_RQ:.6f}  "
              f"(difference {abs(k_RQ - k0):.2e})")

    results = []
    for ds in dsigma_a_values:
        new_mat = _perturbed_material(target_material, dsigma_a=ds, group=group)
        new_regs = _regions_with_material_replaced(base_regions,
                                                    target_material, new_mat)
        new_mesh = Mesh(new_regs)
        A2, F2 = build_matrices(new_mesh)

        # Exact solution of the perturbed problem
        fwd2 = solve_keff(A2, F2, method=method)
        k_pert = fwd2["k"]
        dk_exact = k_pert - k0

        # First-order prediction using the *unperturbed* phi/phi_adj
        dA = (A2 - A).tocsr()
        dF = (F2 - F).tocsr()
        dk_pred = predict_dk_first_order(phi, phi_adj, A, F, dA, dF, k0)

        if verbose:
            print(f"  ds = {ds:+.4e}:  k_pert = {k_pert:.6f}  "
                  f"dk_exact = {dk_exact:+.4e}  "
                  f"dk_predicted = {dk_pred:+.4e}  "
                  f"rel.err = {abs(dk_exact - dk_pred)/max(abs(dk_exact), 1e-30):.2%}")
        results.append(PerturbationResult(ds, k_pert, dk_exact, dk_pred))

    return {
        "base": {"k": k0, "phi": phi, "phi_adj": phi_adj, "A": A, "F": F,
                 "mesh": mesh},
        "results": results,
    }


# ---------------------------------------------------------------------------
def plot_rayleigh_validation(results, save=None, ax=None, title=None):
    """Plot dk_exact vs dk_predicted from a sweep_absorption() output."""
    import matplotlib.pyplot as plt

    ds = np.array([r.dsigma_a for r in results])
    dk_e = np.array([r.dk_exact for r in results])
    dk_p = np.array([r.dk_predicted for r in results])

    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig = ax.figure
        axes = [ax, ax.twinx()]

    axes[0].plot(ds, dk_e, "o-", label=r"Exact $\Delta k$")
    axes[0].plot(ds, dk_p, "s--", label=r"First-order (Rayleigh)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].set_xlabel(r"$\Delta\Sigma_a$  [1/cm]")
    axes[0].set_ylabel(r"$\Delta k$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    if title:
        axes[0].set_title(title)

    if own_fig:
        # Mask the unperturbed point (dk = 0) where the relative error
        # is undefined (and is reported as 100% by the table).
        scale = np.max(np.abs(dk_e)) if dk_e.size else 1.0
        valid = np.abs(dk_e) > 1e-3 * scale
        rel_err = np.full_like(dk_e, np.nan, dtype=float)
        rel_err[valid] = (dk_p[valid] - dk_e[valid]) / dk_e[valid] * 100.0
        axes[1].plot(ds[valid], rel_err[valid], "o-", color="firebrick")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xlabel(r"$\Delta\Sigma_a$  [1/cm]")
        axes[1].set_ylabel("Relative error of 1st-order prediction [%]")
        axes[1].grid(True, alpha=0.3)
        if title:
            axes[1].set_title(title + " - error")

        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150)
    return axes


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from materials import PWR_2G, H2O_2G

    base = Regions(
        Region("LeftRefl",  w=20.0, n=21,  mat=H2O_2G),
        Region("Core",      w=80.0, n=81,  mat=PWR_2G),
        Region("RightRefl", w=20.0, n=21,  mat=H2O_2G),
    )

    print("=" * 60)
    print("Adjoint solve + Rayleigh quotient validation")
    print("=" * 60)
    out = sweep_absorption(
        base_regions=base,
        target_material=PWR_2G,
        group=1,                       # thermal group
        dsigma_a_values=np.linspace(-0.005, 0.005, 9),
        verbose=True,
    )
    plot_rayleigh_validation(
        out["results"],
        save="_test_perturbation.png",
        title="Rayleigh quotient validation: PWR thermal absorption",
    )
    print("\nSaved _test_perturbation.png")
