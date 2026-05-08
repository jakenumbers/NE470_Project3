"""
main.py
=======

Main for NE 470 Project 3.
Reactor Cases:
    Case 1 (DONE) - Bare homogeneous PWR 2-region core (critical-width search)
    Case 2 (DONE) - Homogeneous PWR + water reflector (reflector savings)
    Case 3        - Heterogeneous 5-region 4-group core
    Case 3b       - Layered 5-region core, flat-flux optimisation
    Case 4        - Adjoint + Rayleigh-quotient extra credit
    Case 5        - 4-group homogeneous calculation (extra credit)

"""

#%% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from materials import (
    PWR_2G,
    WATER_2G,
    SS_2G,
    BE_2G,
    C_2G,
    H2O_4G,
    PWR_4G)
from geometry import Region, Regions, Mesh
from matrix_builder import build_matrices
from solver import solve_keff, normalize_to_power_density
from criticality_search import crit_search, reflector_savings
from post_processing import (plot_flux,print_summary,write_flux_csv)
from scipy.optimize import minimize

# Power Density from Small Modular Reactor to Normalize Flux
P_DENSITY_TARGET = 47.0 # W/cm^3   estimated power density

nodes_per_cm = 2.0

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def nodes(width_cm):
    return max(2, int(nodes_per_cm*width_cm)+1)


def _save(name):
    return os.path.join(OUT_DIR, name)

def objective_flatness(phi, x_range):
    """
    Objective function for making the flux profile more flat
    """
    phi_range = phi
    avg_phi = np.average(phi_range)
    max_phi = np.max(phi_range)
    min_phi = np.min(phi_range)
    rmse = np.sqrt(np.mean((phi_range - avg_phi)**2))
    return rmse #/ max_phi

# ---------------------------------------------------------------------------------------
# Case 1: Homogen. PWR, 2-region core
# ---------------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 1: Homogen. PWR, 2-region core")
print("\n" + "=" * 70)
#Crit Search (Core Material, Node Density, Min Width, Max Width, Default Tolerancing is found in python file)
bare_results, bw, mesh_b = crit_search(PWR_2G,nodes_per_cm,5.0,600.0,False)
print(f'Bare Results Final Width: {bw:.3f}')

plot_flux(bare_results["phi"], mesh_b, bare_results['k'],
          save=_save("Bare_Core.png"))
# plot_flux(bare_results["phi"], mesh_b, bare_results['k'],
#           title=f"Bare core (k = {bare_results['k']:.5f})",
#           save=_save("Bare_Core.png"))


# ---------------------------------------------------------------------------------------
# Case 2: Homogen. PWR 2-region core with reflector, savings calc.
# ---------------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 2: PWR, 2-region core with reflector, savings calculated")
print("\n" + "=" * 70)
#Reflector Savings
for ref_width in np.linspace(5,25,5):
    ref_save_results, rsw, mesh_rs = reflector_savings(PWR_2G,WATER_2G,ref_width,2.0,2.0,75.0,False)
    print(f'Reflector Savings Results Final Width: {rsw:.3f}')
    ref_save = (bw - rsw)/2
    print(f'Reflector Savings: {ref_save:.3f}')

plot_flux(ref_save_results["phi"], mesh_rs, ref_save_results['k'],
          save=_save("Refl_Core.png"))

# ---------------------------------------------------------------------------------------
# Case 3a: Heterogeneous 5-region 2-group core
# ---------------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 3a: Heterogeneous 5-region 2-group core")
print("\n" + "=" * 70)
# Design: Burnt Fuel | Moderator | Fresh Fuel | Cladding | Water

regions_3a = Regions(
    Region("Left Water",        w=20, n=nodes(10), mat=WATER_2G),
    Region("Left SS Refl.",     w=5, n=nodes(5), mat=SS_2G),
    Region("Left Fuel, BU10",   w=20, n=nodes(20), mat=PWR_2G),
    Region("Central BeO Refl.",   w=10, n=nodes(30), mat=BE_2G),
    Region("Right Fuel, BU10",  w=20, n=nodes(20), mat=PWR_2G),
    Region("Right SS Refl.",    w=5, n=nodes(5), mat=SS_2G),
    Region("Right Water",       w=20, n=nodes(10), mat=WATER_2G)
)


mesh_3a = Mesh(regions_3a)
A_3a, F_3a = build_matrices(mesh_3a)
res_3a = solve_keff(A_3a, F_3a)
res_3a_normalized = res_3a.copy()
res_3a_normalized["phi"], _ = normalize_to_power_density(res_3a["phi"], mesh_3a, P_DENSITY_TARGET)
# print_summary(res_3a_normalized, mesh_3a)

plot_flux(res_3a_normalized["phi"], mesh_3a, res_3a_normalized['k'],
          save=_save("Case3a_7-reg_2G.png"))



# ---------------------------------------------------------------------------------------
# Case 3b: Heterogeneous 5-region 4-group core
# ---------------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 3: Heterogeneous 5-region 4-group core")
print("\n" + "=" * 70)
regions_3a = Regions(
    Region("Left Water",        w=20, n=nodes(10), mat=H2O_4G),
    Region("Left BeO Refl.",    w=30, n=nodes(30), mat=H2O_4G),
    Region("Left Fuel, BU10",   w=10, n=nodes(20), mat=H2O_4G),
    Region("Central BeO Refl.",   w=10, n=nodes(30), mat=H2O_4G),
    Region("Right Fuel, BU10",  w=10, n=nodes(20), mat=PWR_4G),
    Region("Right SS Refl.",    w=5, n=nodes(5), mat=H2O_4G),
    Region("Right Water",       w=20, n=nodes(10), mat=H2O_4G)
)

mesh_3a = Mesh(regions_3a)
A_3a, F_3a = build_matrices(mesh_3a)
res_3a = solve_keff(A_3a, F_3a)
res_3a_normalized = res_3a.copy()
res_3a_normalized["phi"], _ = normalize_to_power_density(res_3a["phi"], mesh_3a, P_DENSITY_TARGET)
# print_summary(res_3a_normalized, mesh_3a)

plot_flux(res_3a_normalized["phi"], mesh_3a, res_3a_normalized['k'],
          save=_save("Case3a_9-reg_4G.png"))


# ---------------------------------------------------------------------------------------
# Case 3c: Layered 5-region core, flat-flux optimisation
# ---------------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 3: Flat-flux optimization of 5-region 4-group core")
print("\n" + "=" * 70)

widths_initial = [20, 10, 10, 10]

def case3_optimization(widths):
    w1, w2, w3, w4 = widths
    regions_3a = Regions(
        Region("Left Water",        w=w1, n=nodes(w1), mat=H2O_4G),
        Region("Left BeO Refl.",    w=w2, n=nodes(w2), mat=H2O_4G),
        Region("Left Fuel, BU10",   w=w3, n=nodes(w3), mat=H2O_4G),
        Region("Central BeO Refl.",   w=w4, n=nodes(w4), mat=H2O_4G),
        Region("Right Fuel, BU10",  w=w3, n=nodes(w3), mat=PWR_4G),
        Region("Right SS Refl.",    w=w2, n=nodes(w2), mat=H2O_4G),
        Region("Right Water",       w=w1, n=nodes(w1), mat=H2O_4G)
    )

    mesh_3a = Mesh(regions_3a)
    A_3a, F_3a = build_matrices(mesh_3a)
    res_3a = solve_keff(A_3a, F_3a)
    res_3a_normalized = res_3a.copy()
    res_3a_normalized["phi"], _ = normalize_to_power_density(res_3a["phi"], mesh_3a, P_DENSITY_TARGET)
    
    flattness = objective_flatness(res_3a["phi"], mesh_3a)

    return flattness

optimized_widths = minimize(case3_optimization, widths_initial)
print(f'Optimized widths: {optimized_widths}')

w1, w2, w3, w4 = optimized_widths.x
regions_3a = Regions(
    Region("Left Water",        w=w1, n=nodes(w1), mat=H2O_4G),
    Region("Left BeO Refl.",    w=w2, n=nodes(w2), mat=H2O_4G),
    Region("Left Fuel, BU10",   w=w3, n=nodes(w3), mat=H2O_4G),
    Region("Central BeO Refl.",   w=w4, n=nodes(w4), mat=H2O_4G),
    Region("Right Fuel, BU10",  w=w3, n=nodes(w3), mat=PWR_4G),
    Region("Right SS Refl.",    w=w2, n=nodes(w2), mat=H2O_4G),
    Region("Right Water",       w=w1, n=nodes(w1), mat=H2O_4G)
)

mesh_3a = Mesh(regions_3a)
A_3a, F_3a = build_matrices(mesh_3a)
res_3a = solve_keff(A_3a, F_3a)
res_3a_normalized = res_3a.copy()
res_3a_normalized["phi"], _ = normalize_to_power_density(res_3a["phi"], mesh_3a, P_DENSITY_TARGET)
# print_summary(res_3a_normalized, mesh_3a)

plot_flux(res_3a_normalized["phi"], mesh_3a, res_3a_normalized['k'],
          save=_save("Case3a_9-reg_4G.png"))


# ---------------------------------------------------------------------------------------
# Case 4: Adjoint + Rayleigh-quotient extra credit
# ---------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------------
# Case 5: Reflector savings for 4-group
# ---------------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Case 5: Reflector savings problem with 4-group")
print("\n" + "=" * 70)
#Reflector Savings
for ref_width in np.linspace(5,25,5):
    ref_save_results, rsw, mesh_rs = reflector_savings(PWR_4G,H2O_4G,ref_width,2.0,2.0,75.0,False)
    print(f'Reflector Savings Results Final Width: {rsw:.3f}')
    ref_save = (bw - rsw)/2
    print(f'Reflector Savings: {ref_save:.3f}')

plot_flux(ref_save_results["phi"], mesh_rs, ref_save_results['k'],
          save=_save("Refl_Core_4G.png"))




