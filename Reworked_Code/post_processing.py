"""
post_processing.py
==================

Result plotting functions

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from solver import reshape_flux, average_power_density


#%% Region Shading Function
def shade_regions(ax, regions):
    """Shade the axes background by region for context."""
    cmap = plt.get_cmap("Pastel1")
    x_off = 0.0
    for idx, r in enumerate(regions.regions):
        ax.axvspan(x_off, x_off + r.w, color=cmap(idx % cmap.N), alpha=0.35,
                   zorder=0, label=f"{r.name} ({r.mat.name})")
        x_off += r.w


def legend_no_dupes(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), **kw)


#%% Plot Flux
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
        fig, ax = plt.subplots(figsize=(9, 5))

    if show_regions:
        shade_regions(ax, mesh.regions)

    group_labels = ["Fast (g=1)", "Epi (g=2)", "Res (g=3)", "Thermal (g=4)"]
    if G == 2:
        group_labels = ["Fast (g=1)", "Thermal (g=2)"]
    colors = ["C0", "C3", "C2", "C4"]

    for g in range(G):
        lab = group_labels[g] if g < len(group_labels) else f"Group {g+1}"
        ax.plot(mesh.x, phi_gx[g], color=colors[g % len(colors)],
                lw=2, label=lab)

    ax.set_xlabel("Position x [cm]")
    ax.set_ylabel("Flux  $\\phi_g$" + (" (normalised)" if normalize else " [n/cm$^2$/s]"))
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    legend_no_dupes(ax, fontsize=8, loc="best")

    if save and own_fig:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save, dpi=150)
    elif own_fig:
        fig.tight_layout()
    return ax


#%%
def plot_convergence(history, title=None, save=None, ax=None):
    """Plot k_eff vs power-iteration step."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
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


#%% Summary
def print_summary(result, mesh):
    """Print a console summary of a solution."""
    k = result["k"]
    phi = result["phi"]
    iters = result.get("iters")
    conv = result.get("converged")
    print("_" * 60)
    print(" "*60)
    print("Solution summary")
    print("_" * 60)
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


#%%
def write_flux_csv(phi, mesh, path):
    """Save x and per-group flux to CSV."""
    G = mesh.G
    N = mesh.N
    phi_gx = reshape_flux(phi, G, N)
    header = "x_cm," + ",".join(f"phi_g{g+1}" for g in range(G))
    data = np.column_stack([mesh.x, phi_gx.T])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path

