"""
criticality_search.py
=====================

Search for the critical dimension of a homogeneous (or partially layered)
reactor that satisfies ``k_eff = 1``.

Two helpers are provided:

search_critical_width(...)
    Bisection on the *width of a single region* (typically the core).
    A factory function constructs the geometry as a function of that
    width; this allows the same routine to be used for bare cores,
    cores with a fixed reflector thickness, etc.

reflector_savings(...)
    Convenience: compute (bare critical width) - (reflected critical width)
    using the textbook PWR (or any user-supplied) core/reflector
    combination.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from geometry import Region, Regions, Mesh
from matrix_builder import build_matrices
from solver import solve_keff


# ---------------------------------------------------------------------------
def k_for_width(builder: Callable[[float], Regions],
                w: float,
                method: str = "arpack",
                **solver_kwargs) -> float:
    """Build geometry at width ``w``, solve, return k_eff."""
    regions = builder(w)
    mesh = Mesh(regions)
    A, F = build_matrices(mesh)
    res = solve_keff(A, F, method=method, **solver_kwargs)
    return res["k"]


# ---------------------------------------------------------------------------
def search_critical_width(
    builder: Callable[[float], Regions],
    w_low: float,
    w_high: float,
    target_k: float = 1.0,
    tol_w: float = 1e-3,
    tol_k: float = 1e-5,
    max_iter: int = 100,
    method: str = "arpack",
    verbose: bool = True,
):
    """
    Bisection search on the parameter ``w`` of a geometry builder.

    The builder must accept a single positive float (the width) and return
    a :class:`geometry.Regions` instance whose criticality is monotonically
    increasing with ``w``.

    Parameters
    ----------
    w_low, w_high : initial bracket for bisection.  Must satisfy
        k(w_low) < target_k < k(w_high).  If not, the bracket is expanded
        (doubling) up to ``max_iter`` times.

    Returns
    -------
    dict with keys
        w_critical, k_at_w_critical, history (list of (w, k)), bracket
    """
    history = []
    k_lo = k_for_width(builder, w_low, method=method)
    k_hi = k_for_width(builder, w_high, method=method)
    history.append((w_low, k_lo))
    history.append((w_high, k_hi))

    if verbose:
        print(f"  Initial bracket:  w_lo={w_low:.4f} -> k={k_lo:.6f},  "
              f"w_hi={w_high:.4f} -> k={k_hi:.6f}")

    # Expand bracket if necessary
    expand_count = 0
    while (k_lo - target_k) * (k_hi - target_k) > 0 and expand_count < 30:
        if k_hi < target_k:        # need bigger w
            w_high *= 2.0
            k_hi = k_for_width(builder, w_high, method=method)
            history.append((w_high, k_hi))
            if verbose:
                print(f"  Expand high:  w_hi={w_high:.4f} -> k={k_hi:.6f}")
        elif k_lo > target_k:      # need smaller w
            w_low *= 0.5
            k_lo = k_for_width(builder, w_low, method=method)
            history.append((w_low, k_lo))
            if verbose:
                print(f"  Expand low :  w_lo={w_low:.4f} -> k={k_lo:.6f}")
        expand_count += 1

    if (k_lo - target_k) * (k_hi - target_k) > 0:
        raise RuntimeError(
            "Could not bracket k_eff = {:.4f}.  k(w_lo)={:.5f}, "
            "k(w_hi)={:.5f}".format(target_k, k_lo, k_hi))

    # Bisection
    for it in range(max_iter):
        w_mid = 0.5 * (w_low + w_high)
        k_mid = k_for_width(builder, w_mid, method=method)
        history.append((w_mid, k_mid))
        if verbose:
            print(f"  iter {it+1:2d}: w={w_mid:9.4f}  k={k_mid:.6f}  "
                  f"|w_hi - w_lo|={abs(w_high - w_low):.4f}")

        if abs(k_mid - target_k) < tol_k:
            return {
                "w_critical": w_mid,
                "k_at_w_critical": k_mid,
                "history": history,
                "bracket": (w_low, w_high),
            }
        if (k_mid - target_k) * (k_lo - target_k) < 0:
            w_high = w_mid
            k_hi = k_mid
        else:
            w_low = w_mid
            k_lo = k_mid

        if abs(w_high - w_low) < tol_w:
            return {
                "w_critical": 0.5 * (w_low + w_high),
                "k_at_w_critical": k_mid,
                "history": history,
                "bracket": (w_low, w_high),
            }

    return {
        "w_critical": 0.5 * (w_low + w_high),
        "k_at_w_critical": k_mid,
        "history": history,
        "bracket": (w_low, w_high),
    }


# ---------------------------------------------------------------------------
def reflector_savings(
    core_mat,
    refl_mat=None,
    refl_thickness: float = 0.0,
    n_per_cm: float = 0.25,
    w_low: float = 10.0,
    w_high: float = 1000.0,
    method: str = "arpack",
    verbose: bool = True,
):
    """
    Compute the reflector savings: (bare critical width) - (reflected
    critical width).

    Parameters
    ----------
    core_mat : Material
        Material of the core.
    refl_mat : Material or None
        Material of the reflector.  If None, only the bare problem is
        solved.
    refl_thickness : float
        Width of each reflector (left and right) in cm.
    n_per_cm : float
        Mesh density (nodes per cm) used in the geometry.
    """
    def _bare_builder(w):
        n = max(2, int(n_per_cm * w) + 1)
        return Regions(Region("Core", w=w, n=n, mat=core_mat))

    if verbose:
        print("\n--- Bare critical width ---")
    bare = search_critical_width(_bare_builder, w_low, w_high,
                                 method=method, verbose=verbose)
    if verbose:
        print(f"\nBare critical width = {bare['w_critical']:.4f} cm "
              f"(k = {bare['k_at_w_critical']:.6f})")

    if refl_mat is None or refl_thickness <= 0.0:
        return {
            "bare_w": bare["w_critical"],
            "reflected_w": None,
            "savings": None,
            "bare_history": bare["history"],
            "reflected_history": None,
        }

    n_refl = max(2, int(n_per_cm * refl_thickness) + 1)

    def _refl_builder(w):
        n = max(2, int(n_per_cm * w) + 1)
        return Regions(
            Region("LeftRefl",  w=refl_thickness, n=n_refl, mat=refl_mat),
            Region("Core",      w=w,              n=n,      mat=core_mat),
            Region("RightRefl", w=refl_thickness, n=n_refl, mat=refl_mat),
        )

    if verbose:
        print(f"\n--- Critical width with {refl_mat.name} reflector "
              f"({refl_thickness} cm each side) ---")
    refl = search_critical_width(_refl_builder, w_low, w_high,
                                 method=method, verbose=verbose)
    if verbose:
        print(f"\nReflected critical width = {refl['w_critical']:.4f} cm "
              f"(k = {refl['k_at_w_critical']:.6f})")
        print(f"Reflector savings = "
              f"{bare['w_critical'] - refl['w_critical']:.4f} cm "
              f"(per side: {(bare['w_critical'] - refl['w_critical'])/2:.4f} cm)")

    return {
        "bare_w":      bare["w_critical"],
        "reflected_w": refl["w_critical"],
        "savings":     bare["w_critical"] - refl["w_critical"],
        "bare_history":      bare["history"],
        "reflected_history": refl["history"],
    }


if __name__ == "__main__":
    from materials import PWR_2G, H2O_2G

    print("=" * 60)
    print("Bare PWR core critical width search")
    print("=" * 60)
    info = reflector_savings(
        core_mat=PWR_2G,
        refl_mat=H2O_2G,
        refl_thickness=20.0,
        n_per_cm=0.25,
        w_low=20.0,
        w_high=400.0,
        verbose=True,
    )
    print()
    print(f"Bare critical width      : {info['bare_w']:.3f} cm")
    print(f"Reflected critical width : {info['reflected_w']:.3f} cm")
    print(f"Reflector savings (total): {info['savings']:.3f} cm")
    print(f"Reflector savings (per side): {info['savings']/2:.3f} cm")
