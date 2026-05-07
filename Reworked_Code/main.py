"""
main.py
=======

Main for NE 470 Project 3.
Reactor Cases:
    Case 1   - Bare homogeneous PWR 2-region core (critical-width search)
    Case 1b  - Bare homogeneous PWR 4-region core (critical-width search)
    Case 2   - Homogeneous PWR + water reflector (reflector savings)
    Case 3   - Heterogeneous 5-region 4-group core
    Case 3b  - Layered 5-region core, flat-flux optimisation
    Case 4   - Adjoint + Rayleigh-quotient extra credit
    Case 5   - 4-group homogeneous calculation (extra credit)

"""

#%% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from materials import (
    PWR_2G,
    WATER_2G)
from geometry import Region, Regions, Mesh
from matrix_builder import build_matrices
from solver import solve_keff, normalize_to_power_density
from criticality_search import crit_search, reflector_savings
from post_processing import (plot_flux,print_summary,write_flux_csv)

# Power Density from Small Modular Reactor to Normalize Flux
P_DENSITY_TARGET = 47.0 # W/cm^3   estimated power density

nodes_per_cm = 2.0

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(name):
    return os.path.join(OUT_DIR, name)

#Crit Search (Core Material, Node Density, Min Width, Max Width, Default Tolerancing is found in python file)
bare_results, bw, mesh_b = crit_search(PWR_2G,nodes_per_cm,5.0,600.0,False)
print(f'Bare Results Final Width: {bw:.3f}')

plot_flux(bare_results["phi"], mesh_b,
          title=f"Bare core (k = {bare_results['k']:.5f})",
          save=_save("Bare_Core.png"))

#Reflector Savings
for ref_width in np.linspace(5,25,5):
    ref_save_results, rsw, mesh_rs = reflector_savings(PWR_2G,WATER_2G,ref_width,2.0,2.0,75.0,False)
    print(f'Reflector Savings Results Final Width: {rsw:.3f}')
    ref_save = (bw - rsw)/2
    print(f'Reflector Savings: {ref_save:.3f}')

plot_flux(ref_save_results["phi"], mesh_rs,
          title=f"Reflector + core (k = {ref_save_results['k']:.5f})",
          save=_save("Refl_Core.png"))




