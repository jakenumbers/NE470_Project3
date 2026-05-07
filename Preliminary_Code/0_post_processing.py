"""
post_processing.py
==================

Plotting and tabular output utilities for the NE 470 Project 3 solver.

All functions take the *flat* group-major flux vector ``phi`` returned by
:func:`solver.solve_keff`, plus the :class:`geometry.Mesh` it was solved on.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from solver import reshape_flux, average_power_density, _node_kappa_sigma_f

# ---------------------------------------------------------------------------
"""
Plot styles, according to IEEE
"""
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 7.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def _shade_regions(ax, regions):
    """Shade the axes background by region for context."""
    cmap = plt.get_cmap("Pastel1")
    x_off = 0.0
    for idx, r in enumerate(regions.regions):
        ax.axvspan(x_off, x_off + r.w, color=cmap(idx % cmap.N), alpha=0.35,
                   zorder=0, label=f"{r.name} ({r.mat.name})")
        x_off += r.w


def _legend_no_dupes(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), **kw)


# ---------------------------------------------------------------------------
def plot_flux(
    phi,
    mesh,
    title=None,
    normalize=True,
    show_regions=True,
    save=None,
    ax=None,
):
    """Plot per-group flux vs. position.

    Parameters
    ----------
    phi : flat group-major flux vector
    mesh : Mesh object
    normalize : if True, scale each group so that max(group 0) = 1
                (purely cosmetic if the flux has been power-normalised)
    show_regions : shade background by region
    save : optional path to save figure (PNG/PDF)
    ax : existing matplotlib axes; if None, create a new figure
    """
    G = mesh.G
    N = mesh.N
    phi_gx = reshape_flux(phi, G, N)

    if normalize:
        scale = np.abs(phi_gx).max() or 1.0
        phi_gx = phi_gx / scale

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))

    if show_regions:
        _shade_regions(ax, mesh.regions)

    group_labels = ["Fast (g=1)", "Epi (g=2)", "Res (g=3)", "Thermal (g=4)"]
    if G == 2:
        group_labels = ["Fast (g=1)", "Thermal (g=2)"]
    colors = ["C0", "C3", "C2", "C4"]

    for g in range(G):
        lab = group_labels[g] if g < len(group_labels) else f"Group {g+1}"
        ax.plot(mesh.x, phi_gx[g], color=colors[g % len(colors)],
                lw=2, label=lab)

    ax.set_xlabel("Position x [cm]")
    ax.set_ylabel("Flux  $\\phi_g$" + (" (normalized)" if normalize else " [n/cm$^2$/s]"))
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _legend_no_dupes(ax, fontsize=8, loc="best")

    if save and own_fig:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save, dpi=150)
    elif own_fig:
        fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
def plot_power_density(
    phi,
    mesh,
    title=None,
    show_regions=True,
    save=None,
    ax=None,
):
    """Plot kappa-Sigma_f * phi summed over groups (power density per cm)."""
    G = mesh.G
    N = mesh.N
    phi_gx = reshape_flux(phi, G, N)

    p = np.zeros(N)
    for i in range(N):
        for g in range(G):
            kf_avg = 0.0
            wsum = 0.0
            if mesh.mat_left[i] is not None:
                kf_avg += _node_kappa_sigma_f(mesh.mat_left[i], g) * mesh.dx_left[i] * 0.5
                wsum += mesh.dx_left[i] * 0.5
            if mesh.mat_right[i] is not None:
                kf_avg += _node_kappa_sigma_f(mesh.mat_right[i], g) * mesh.dx_right[i] * 0.5
                wsum += mesh.dx_right[i] * 0.5
            if wsum > 0:
                p[i] += (kf_avg / wsum) * phi_gx[g, i]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))
    if show_regions:
        _shade_regions(ax, mesh.regions)
    ax.plot(mesh.x, p, color="firebrick", lw=2, label=r"$\sum_g \kappa\Sigma_{f,g}\phi_g$")
    ax.set_xlabel("Position x [cm]")
    ax.set_ylabel("Power density [W/cm$^3$]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _legend_no_dupes(ax, fontsize=8)
    if save and own_fig:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save, dpi=150)
    elif own_fig:
        fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
def plot_convergence(history, title=None, save=None, ax=None):
    """Plot k_eff vs power-iteration step."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, "o-", ms=3)
    ax.set_xlabel("Power iteration")
    ax.set_ylabel("$k$")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if save and own_fig:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save, dpi=150)
    elif own_fig:
        fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
def print_summary(result, mesh):
    """Print a console summary of a solution."""
    k = result["k"]
    phi = result["phi"]
    iters = result.get("iters")
    conv = result.get("converged")
    print("=" * 60)
    print("Solution summary")
    print("=" * 60)
    print(mesh.summary())
    print(f"\nk_eff = {k:.6f}")
    if iters is not None:
        print(f"Power iterations: {iters} (converged = {conv})")
    try:
        p_avg = average_power_density(phi, mesh)
        print(f"Average power density (fuel): {p_avg:.3f} W/cm^3")
    except RuntimeError:
        pass
    G = mesh.G
    N = mesh.N
    phi_gx = reshape_flux(phi, G, N)
    print("\nPeak flux per group:")
    for g in range(G):
        ipeak = int(np.argmax(np.abs(phi_gx[g])))
        print(f"  group {g+1}: phi_max = {phi_gx[g, ipeak]:.4e} "
              f"at x = {mesh.x[ipeak]:.2f} cm")
    print()


# ---------------------------------------------------------------------------
def write_flux_csv(phi, mesh, path):
    """Save x and per-group flux to CSV."""
    G = mesh.G
    N = mesh.N
    phi_gx = reshape_flux(phi, G, N)
    header = "x_cm," + ",".join(f"phi_g{g+1}" for g in range(G))
    data = np.column_stack([mesh.x, phi_gx.T])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from materials import PWR_2G, H2O_2G
    from geometry import Region, Regions, Mesh
    from matrix_builder import build_matrices
    from solver import solve_keff, normalize_to_power_density

    regs = Regions(
        Region("LeftRefl",  w=30.0,  n=31,  mat=H2O_2G),
        Region("Core",      w=300.0, n=151, mat=PWR_2G),
        Region("RightRefl", w=30.0,  n=31,  mat=H2O_2G),
    )
    m = Mesh(regs)
    A, F = build_matrices(m)
    res = solve_keff(A, F, method="arpack")
    res["phi"], _ = normalize_to_power_density(res["phi"], m,
                                               target_W_per_cm3=47.0)
    print_summary(res, m)
    plot_flux(res["phi"], m,
              title=f"PWR + H2O reflectors (k={res['k']:.5f})",
              save="_test_flux.png")
    plot_power_density(res["phi"], m, title="Power density",
                       save="_test_power.png")
    print("Saved _test_flux.png, _test_power.png")
