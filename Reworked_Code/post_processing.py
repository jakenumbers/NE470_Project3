"""
post_processing.py
==================

Result plotting functions

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from solver import reshape_flux, average_power_density
import math
import seaborn as sns
import matplotlib.patches as mpatches

"""
Plot styles, according to IEEE
"""
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 4,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

#%% Region Shading Function
def shade_regions(ax, regions, max_phi):
    """Shade the axes background by region for context."""
    x_off = 0.0
    num_regions = len(regions.regions)
    colors = []
    if num_regions % 2 == 0:
        num_colors = math.ceil(num_regions/2)
        colors = sns.color_palette("pastel", num_colors)
        colors.extend(colors[::-1])
    else:
        if num_regions == 1:
            num_colors = num_regions
        else:
            num_colors = math.ceil(num_regions/2)
        colors = sns.color_palette("pastel", num_colors)
        colors += colors[::-1][1:]
    hatch_styles = []
    for i in range(num_regions):
        if num_regions % 2 == 0:
            if i == math.floor((num_regions-1)/2):
                hatch_styles.append('//')
            elif i == math.ceil((num_regions-1)/2):
                hatch_styles.append('\\\\')
            elif i % 2 == 0 and i > ((num_regions-1)/2):
                hatch_styles.append('\\\\')
            elif i % 2 == 1 and i < ((num_regions-1)/2):
                hatch_styles.append('//')
            elif i % 2 == 1 and i > ((num_regions-1)/2):
                hatch_styles.append('//')
            elif i % 2 == 0 and i < ((num_regions-1)/2):
                hatch_styles.append('\\\\')
        else:
            if i == ((num_regions-1)/2):
                hatch_styles.append('')
            elif i % 2 == 1 and i > ((num_regions-1)/2):
                hatch_styles.append('\\\\')
            elif i % 2 == 1 and i < ((num_regions-1)/2):
                hatch_styles.append('//')
            elif i % 2 == 0 and i > ((num_regions-1)/2):
                hatch_styles.append('//')
            elif i % 2 == 0 and i < ((num_regions-1)/2):
                hatch_styles.append('\\\\')
    colors.extend(colors[::-1]) # Reverse list and append it

    print('Widths:')
    for idx, r in enumerate(regions.regions):
        # ax.axvspan(x_off, x_off + r.w, color=cmap(idx % cmap.N), alpha=0.35, zorder=0, label=f"{r.name} ({r.mat.name})")
        ax.fill_between([x_off, (x_off + r.w)], [max_phi*1.1,max_phi*1.1], hatch=hatch_styles[idx], facecolor=(colors[idx],0.35), 
                        edgecolor=('gray',0.1), zorder=0, 
                        label=f"{r.mat.name} (${r.w:.1f}$ cm)")#label=f"{r.mat.name} ($W_\mathrm{{{r.name}}} = {r.w:.1f}$ cm)")
        x_off += r.w
        print(f'   {r.name}: {r.w}')


def legend_no_dupes(ax,k_val, **kw):
    handles, labels = ax.get_legend_handles_labels()
    k_patch = mpatches.Patch(color='none',label=f"k = {k_val:.5f}")
    handles.append(k_patch)
    labels.append(f"k = {k_val:.5f}")
    for patch in ax.patches:
        patch.set_hatch('')
    seen = {}
    for h, l in zip(handles, labels):
        # if l not in seen:
        seen[l] = h
    leg = ax.legend(seen.values(), seen.keys(), **kw)
    


#%% Plot Flux
def plot_flux(
    phi,
    mesh,
    k,
    title=None,
    normalize=False,
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
                (purely cosmetic if the flux has been power-Normalized)
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
        fig, ax = plt.subplots(figsize=(5, 4))

    if show_regions:
        shade_regions(ax, mesh.regions, np.max(phi_gx))

    group_labels = ["Fast (g=1)", "Epithermal (g=2)", "Resonance (g=3)", "Thermal (g=4)"]
    if G == 2:
        group_labels = ["Fast (g=1)", "Thermal (g=2)"]
    colors = ["C0", "C3", "C2", "C4"]

    for g in range(G):
        lab = group_labels[g] if g < len(group_labels) else f"Group {g+1}"
        ax.plot(mesh.x, phi_gx[g], color=colors[g % len(colors)],
                lw=2, label=lab)

    ax.set_xlabel("Position x [cm]")
    ax.set_ylabel("Flux  $\\phi_g$" + (" (Normalized)" if normalize else " [n/cm$^2$/s]"))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    if title:
        # ax.set_title(title)
        ax.text(0.05,0.94, title, transform=ax.transAxes, fontsize=12, fontweight='semibold', va='top', ha='left')
    # ax.grid(True, alpha=0.3)
    legend_no_dupes(ax, k, loc="best", fontsize=8.5)

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
        fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(history, "o-", ms=3)
    ax.set_xlabel("Power iteration")
    ax.set_ylabel("$k$")
    # ax.grid(True, alpha=0.3)
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

