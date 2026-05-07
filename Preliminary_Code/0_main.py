"""
main.py
=======

Driver script for NE 470 Project 3.

Layout: Spyder ``#%%`` cells.  Open in Spyder and run cells with
Ctrl+Enter (Run Cell) / Shift+Enter (Run Cell and Advance), or run
the whole script with ``python main.py`` to execute every case in
order.

Cases:
    Case 1   - Bare homogeneous PWR 2-group core (critical-width search)
    Case 1b   - Bare homogeneous PWR 4-group core (critical-width search)
    Case 2   - Homogeneous PWR 2-group + water reflector (reflector savings)
    Case 3   - Heterogeneous 5-region 4-group core
    Case 3b  - Layered 5-region core, flat-flux optimisation
    Case 4   - Adjoint + Rayleigh-quotient extra credit
    Case 5   - 4-group homogeneous calculation (extra credit)

Per-case variables are suffix ed (``_c1``, ``_c2``, ...) so running
cells in any order does not clobber earlier results.
"""

#%% Imports and configuration
import os
import numpy as np
import matplotlib.pyplot as plt

from materials import (
    PWR_2G,
    H2O_2G, H2O_4G,
    UO2_W17_FRESH_4G, UO2_W17_BU10_4G, UO2_W17_BU30_4G,
)

from geometry import Region, Regions, Mesh
from matrix_builder import build_matrices
from solver import solve_keff, normalize_to_power_density
from post_processing import (
    plot_flux,
    plot_power_density,
    print_summary,
    write_flux_csv,
)
from criticality_search import reflector_savings
from perturbation import sweep_absorption, plot_rayleigh_validation

# Flux is normalised so the volume-averaged power density across the
# fuel-bearing cells matches a representative PWR fuel value.
P_DENSITY_TARGET = 47.0           # W/cm^3   estimated PWR fuel power density

# Mesh density used for both the criticality search and the final
# plotted/reported solution.  At 2 nodes/cm the bare PWR critical width
# is converged to ~0.02 % of the 8-nodes/cm reference (25.376 cm vs.
# 25.380 cm), and a 25 cm core has ~50 nodes for smooth plotting.
NODES_PER_CM = 2.0

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(name):
    return os.path.join(OUT_DIR, name)


def _nodes(width_cm):
    """Number of mesh nodes spanning ``width_cm`` at NODES_PER_CM density."""
    return max(2, int(NODES_PER_CM * width_cm) + 1)


#%% Case 1: Bare homogeneous PWR core - critical-width search
print("\n" + "=" * 70)
print("Case 1: Bare homogeneous PWR core - critical-width search")
print("=" * 70)

info_c1 = reflector_savings(
    core_mat=PWR_2G,
    refl_mat=None,
    n_per_cm=NODES_PER_CM,
    w_low=10.0,
    w_high=400.0,
    verbose=True,
)
w_crit_c1 = info_c1["bare_w"]

regs_c1 = Regions(Region("Core", w=w_crit_c1, n=_nodes(w_crit_c1), mat=PWR_2G))
mesh_c1 = Mesh(regs_c1)
A_c1, F_c1 = build_matrices(mesh_c1)
res_c1 = solve_keff(A_c1, F_c1, method="arpack")
res_c1["phi"], _ = normalize_to_power_density(
    res_c1["phi"], mesh_c1, target_W_per_cm3=P_DENSITY_TARGET
)

print_summary(res_c1, mesh_c1)
plot_flux(res_c1["phi"], mesh_c1,
          title=f"Case 1 - Bare PWR (w = {w_crit_c1:.2f} cm, k = {res_c1['k']:.5f})",
          save=_save("case1_bare_flux.pdf"))
write_flux_csv(res_c1["phi"], mesh_c1, _save("case1_bare_flux.csv"))
plt.close("all")


#%% Case 1b: Bare homogeneous PWR 4-group core - critical-width search
print("\n" + "=" * 70)
print("Case 1b: Bare homogeneous PWR 4-group core - critical-width search")
print("=" * 70)

info_c1b = reflector_savings(
    core_mat=UO2_W17_FRESH_4G,
    refl_mat=None,
    n_per_cm=NODES_PER_CM,
    w_low=10.0,
    w_high=400.0,
    verbose=True,
)
w_crit_c1b = info_c1b["bare_w"]

regs_c1b = Regions(Region("Core", w=w_crit_c1b, n=_nodes(w_crit_c1b), mat=UO2_W17_FRESH_4G))
mesh_c1b = Mesh(regs_c1b)
A_c1b, F_c1b = build_matrices(mesh_c1b)
res_c1b = solve_keff(A_c1b, F_c1b, method="arpack")
res_c1b["phi"], _ = normalize_to_power_density(
    res_c1b["phi"], mesh_c1b, target_W_per_cm3=P_DENSITY_TARGET
)

print_summary(res_c1b, mesh_c1b)
plot_flux(res_c1b["phi"], mesh_c1b,
          title=f"Case 1b - Bare PWR 4-group (w = {w_crit_c1b:.2f} cm, k = {res_c1b['k']:.5f})",
          save=_save("case1b_bare_flux.pdf"))
write_flux_csv(res_c1b["phi"], mesh_c1b, _save("case1b_bare_flux.csv"))
plt.close("all")


#%% Case 2: PWR + water reflector - reflector savings
print("\n" + "=" * 70)
print("Case 2: PWR + water reflector - reflector savings")
print("=" * 70)

REFL_T_C2 = 20.0       # cm reflector thickness on each side
info_c2 = reflector_savings(
    core_mat=PWR_2G,
    refl_mat=H2O_2G,
    refl_thickness=REFL_T_C2,
    n_per_cm=NODES_PER_CM,
    w_low=10.0,
    w_high=400.0,
    verbose=True,
)

w_crit_c2 = info_c2["reflected_w"]
regs_c2 = Regions(
    Region("LeftRefl",  w=REFL_T_C2,  n=_nodes(REFL_T_C2),  mat=H2O_2G),
    Region("Core",      w=w_crit_c2,  n=_nodes(w_crit_c2),  mat=PWR_2G),
    Region("RightRefl", w=REFL_T_C2,  n=_nodes(REFL_T_C2),  mat=H2O_2G),
)
mesh_c2 = Mesh(regs_c2)
A_c2, F_c2 = build_matrices(mesh_c2)
res_c2 = solve_keff(A_c2, F_c2, method="arpack")
res_c2["phi"], _ = normalize_to_power_density(
    res_c2["phi"], mesh_c2, target_W_per_cm3=P_DENSITY_TARGET
)

print_summary(res_c2, mesh_c2)
plot_flux(res_c2["phi"], mesh_c2,
          title=(f"Case 2 - Reflected PWR (w$_{{core}}$ = {w_crit_c2:.2f} cm, "
                 f"w$_{{ref}}$ = {REFL_T_C2:.2f} cm, "
                 f"k = {res_c2['k']:.5f})"),
          save=_save("case2_reflected_flux.pdf"))
plot_power_density(res_c2["phi"], mesh_c2,
                   title="Case 2 - Power density",
                   save=_save("case2_reflected_power.pdf"))
write_flux_csv(res_c2["phi"], mesh_c2, _save("case2_reflected_flux.csv"))
plt.close("all")


#%% Case 3: Heterogeneous 5-region 4-group core
# Layout (left -> right):
#   H2O reflector | UO2 (fresh) | UO2 (BU=10) | UO2 (BU=30) | H2O reflector
print("\n" + "=" * 70)
print("Case 3: Heterogeneous 5-region core (4-group)")
print("=" * 70)

regs_c3 = Regions(
    Region("LeftRefl",   w=20.0, n=_nodes(20.0), mat=H2O_4G),
    Region("Fresh",      w=60.0, n=_nodes(60.0), mat=UO2_W17_FRESH_4G),
    Region("MidBurnup",  w=80.0, n=_nodes(80.0), mat=UO2_W17_BU10_4G),
    Region("HighBurnup", w=60.0, n=_nodes(60.0), mat=UO2_W17_BU30_4G),
    Region("RightRefl",  w=20.0, n=_nodes(20.0), mat=H2O_4G),
)
mesh_c3 = Mesh(regs_c3)
A_c3, F_c3 = build_matrices(mesh_c3)
res_c3 = solve_keff(A_c3, F_c3, method="arpack")
res_c3["phi"], _ = normalize_to_power_density(
    res_c3["phi"], mesh_c3, target_W_per_cm3=P_DENSITY_TARGET
)

print_summary(res_c3, mesh_c3)
plot_flux(res_c3["phi"], mesh_c3,
          title=f"Case 3 - 5-region 4-group core (k = {res_c3['k']:.5f})",
          save=_save("case3_5region_flux.pdf"))
plot_power_density(res_c3["phi"], mesh_c3,
                   title="Case 3 - Power density",
                   save=_save("case3_5region_power.pdf"))
write_flux_csv(res_c3["phi"], mesh_c3, _save("case3_5region_flux.csv"))
plt.close("all")


#%% Case 3b: Layered 5-region core - flat-flux optimisation
# Strategy: place fresher (more reactive) fuel near the boundaries and
# higher-burnup fuel at the centre to flatten the radial flux profile.
print("\n" + "=" * 70)
print("Case 3b: Layered 5-region core - flat-flux optimisation")
print("=" * 70)

regs_c3b = Regions(
    Region("LeftRefl",    w=20.0, n=_nodes(20.0), mat=H2O_4G),
    Region("OuterFresh",  w=40.0, n=_nodes(40.0), mat=UO2_W17_FRESH_4G),
    Region("InnerBurnt",  w=80.0, n=_nodes(80.0), mat=UO2_W17_BU30_4G),
    Region("OuterFresh2", w=40.0, n=_nodes(40.0), mat=UO2_W17_FRESH_4G),
    Region("RightRefl",   w=20.0, n=_nodes(20.0), mat=H2O_4G),
)
mesh_c3b = Mesh(regs_c3b)
A_c3b, F_c3b = build_matrices(mesh_c3b)
res_c3b = solve_keff(A_c3b, F_c3b, method="arpack")
res_c3b["phi"], _ = normalize_to_power_density(
    res_c3b["phi"], mesh_c3b, target_W_per_cm3=P_DENSITY_TARGET
)

print_summary(res_c3b, mesh_c3b)
plot_flux(res_c3b["phi"], mesh_c3b,
          title=f"Case 3b - Flat-flux core (k = {res_c3b['k']:.5f})",
          save=_save("case3b_flat_flux.pdf"))
plot_power_density(res_c3b["phi"], mesh_c3b,
                   title="Case 3b - Power density",
                   save=_save("case3b_flat_power.pdf"))
write_flux_csv(res_c3b["phi"], mesh_c3b, _save("case3b_flat_flux.csv"))
plt.close("all")


#%% Case 4: Adjoint + Rayleigh-quotient validation (extra credit)
print("\n" + "=" * 70)
print("Case 4: Adjoint + Rayleigh-quotient (extra credit)")
print("=" * 70)

base_c4 = Regions(
    Region("LeftRefl",  w=20.0, n=_nodes(20.0), mat=H2O_2G),
    Region("Core",      w=80.0, n=_nodes(80.0), mat=PWR_2G),
    Region("RightRefl", w=20.0, n=_nodes(20.0), mat=H2O_2G),
)

out_c4 = sweep_absorption(
    base_regions=base_c4,
    target_material=PWR_2G,
    group=1,                                   # thermal group
    dsigma_a_values=np.linspace(-0.01, 0.01, 11),
    verbose=True,
)

base_info_c4 = out_c4["base"]
mesh_c4 = base_info_c4["mesh"]
fig_c4, axes_c4 = plt.subplots(2, 1, figsize=(6, 4))
plot_flux(base_info_c4["phi"], mesh_c4,
          title=f"Forward flux  (k = {base_info_c4['k']:.5f})",
          ax=axes_c4[0])
plot_flux(base_info_c4["phi_adj"], mesh_c4,
          title="Adjoint flux  $\\phi^*$",
          ax=axes_c4[1])
fig_c4.tight_layout()
fig_c4.savefig(_save("case4_forward_adjoint.pdf"), dpi=150)
plt.close(fig_c4)

plot_rayleigh_validation(
    out_c4["results"],
    save=_save("case4_rayleigh_validation.pdf"),
    title="Case 4 - Rayleigh validation: PWR thermal absorption",
)
plt.close("all")


#%% Case 5: 4-group homogeneous UO2 core (extra credit)
print("\n" + "=" * 70)
print("Case 5: 4-group homogeneous UO2 core (extra credit)")
print("=" * 70)

regs_c5 = Regions(
    Region("LeftRefl",  w=20.0,  n=_nodes(20.0),  mat=H2O_4G),
    Region("Core",      w=200.0, n=_nodes(200.0), mat=UO2_W17_FRESH_4G),
    Region("RightRefl", w=20.0,  n=_nodes(20.0),  mat=H2O_4G),
)
mesh_c5 = Mesh(regs_c5)
A_c5, F_c5 = build_matrices(mesh_c5)
res_c5 = solve_keff(A_c5, F_c5, method="arpack")
res_c5["phi"], _ = normalize_to_power_density(
    res_c5["phi"], mesh_c5, target_W_per_cm3=P_DENSITY_TARGET
)

print_summary(res_c5, mesh_c5)
plot_flux(res_c5["phi"], mesh_c5,
          title=f"Case 5 - 4-group UO2 + H2O (k = {res_c5['k']:.5f})",
          save=_save("case5_4group_flux.pdf"))
write_flux_csv(res_c5["phi"], mesh_c5, _save("case5_4group_flux.csv"))
plt.close("all")
