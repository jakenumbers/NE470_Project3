"""
perturbation.py

"""
import os
import copy
import numpy as np
import scipy.sparse as sp
from materials import Material
from geometry import Region, Regions, Mesh
from matrix_builder import build_matrices
from solver import solve_keff, solve_adjoint, normalize_to_power_density
import matplotlib.pyplot as plt
from post_processing import (plot_flux,print_summary,write_flux_csv)

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
    "legend.fontsize": 7.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def _save(name):
    return os.path.join(OUT_DIR, name)

#%%Rayleigh Quotient Calculation
def rayleigh_calc(phi, phi_adjoint, A, F):
    A_phi = A @ phi
    F_phi = F @ phi
    num = float(np.dot(phi_adjoint, F_phi))
    dem = float(np.dot(phi_adjoint, A_phi))
    return num/dem

#%%Perturbation Calculation
def perturbation_approx(phi, phi_adjoint, F, dF, dA):
    return np.dot(phi_adjoint, (dF- dA) @ phi)/ np.dot(phi_adjoint, F @ phi)

#%% Begin Perturbation from Reactor Geometry
def perturbation_example(regions, mat):
    #Initial Reactor Solved with Adjoint
    mesh = Mesh(regions)
    A, F = build_matrices(mesh)
    forward = solve_keff(A, F)
    adjoint = solve_adjoint(A, F)

    forward_normalized = forward.copy()
    forward_normalized["phi"], _ = normalize_to_power_density(forward["phi"], mesh, 45)
    l_w = np.sqrt(300e6 / (mesh.regions.total_width * 45))
    print(f'Depth and height of slab: {l_w:.3f} cm')
    print(f'Thickness-to-height/depth ratio: {l_w/mesh.regions.total_width:.2f}')
    # print_summary(forward_normalized, mesh_3a)

    plot_flux(forward_normalized["phi"], mesh, forward_normalized['k'], title='Forward Solution',
            save=_save("Case4_forward.pdf"))
    plot_flux(forward_normalized["phi"], mesh, forward_normalized['k'], title='Forward Solution',
            save=_save("Case4_forward.png"))

    adjoint_normalized = adjoint.copy()
    adjoint_normalized["phi"], _ = normalize_to_power_density(adjoint["phi"], mesh, 45)
    l_w = np.sqrt(300e6 / (mesh.regions.total_width * 45))
    print(f'Depth and height of slab: {l_w:.3f} cm')
    print(f'Thickness-to-height/depth ratio: {l_w/mesh.regions.total_width:.2f}')
    # print_summary(adjoint_normalized, mesh_3a)

    plot_flux(adjoint_normalized["phi"], mesh, adjoint_normalized['k'], title='Adjoint Solution',
            save=_save("Case4_adjoint.pdf"))
    plot_flux(adjoint_normalized["phi"], mesh, adjoint_normalized['k'], title='Adjoint Solution',
            save=_save("Case4_adjoint.png"))

    
    #Sets values for Rayleigh Quotient
    k_0, phi, phi_adjoint = forward['k'], forward['phi'], adjoint['phi']
    
    #Check the Rayleigh Quotient is right
    check_k = rayleigh_calc(phi, phi_adjoint, A, F)
    if abs(k_0 - check_k) > 1e-6:
        raise Exception("Something is wrong with the adjoint/transpose of the matrices")
        
    #Initial Reactivity
    rho_0 = 1.0 - 1.0/k_0
    
    #Perturb delta list
    group = 3
    mat_sig_a = mat.sigma_a[group]
    perturb_list = [mat_sig_a*-0.25,mat_sig_a*-0.10,mat_sig_a*-0.05,mat_sig_a*-0.01, mat_sig_a*0.0,
                    mat_sig_a*0.01, mat_sig_a*0.05,mat_sig_a*0.1,mat_sig_a*0.25]
    pred_rho = []
    exact_rho = []
    for val in perturb_list:
        new_mat = perturb_mat(mat, val, group)
        new_regions = rep_mat(regions, mat, new_mat)
        new_mesh = Mesh(new_regions)
        nA, nF = build_matrices(new_mesh)
        new_for = solve_keff(nA, nF)
        exact_k = new_for['k']
        exact_rho.append((1.0 - 1.0/exact_k) - rho_0)
        dA = nA - A
        dF = nF - F
        new_k = perturbation_approx(phi, phi_adjoint, F, dF, dA)
        pred_rho.append(new_k)
    
    x = [-0.25,-0.1,-0.05,-0.01,0,0.01,0.05,0.1,0.25]
    plt.figure(figsize=(5,3))
    plt.plot(x,pred_rho, linestyle='--', label = 'First-order perturbation (Rayleigh)')
    plt.plot(x,exact_rho, label = 'Exact $\Delta k$')
    plt.xlabel('$\Delta \Sigma_a$')
    plt.ylabel('$\Delta$k')
    # plt.title('$\Delta$k')
    plt.legend()
    plt.tight_layout()
    plt.savefig(_save('Case4_perturbation.pdf'))
    plt.savefig(_save('Case4_perturbation.png'))
#%% Changes material properties by a delta
def perturb_mat(mat, dsigma_a, group):
    #copy the material
    new_mat = copy.deepcopy(mat)
    #Changes material's sigma a and r 
    delta = np.zeros_like(new_mat.sigma_a)
    delta[group] = dsigma_a
    new_mat.sigma_a = new_mat.sigma_a + delta
    new_mat.sigma_r = new_mat.sigma_r + delta
    return new_mat
#%% Takes new material and replaces old one in a new regions instance
def rep_mat(regions, target_mat, new_mat):
    perturb_regions = []
    for r in regions.regions:
        if r.mat is target_mat:
            perturb_regions.append(Region(name=r.name, w=r.w, n=r.n, mat=new_mat))
        else:
            perturb_regions.append(r)
    return Regions(*perturb_regions)

#Logic Steps
#Step 1 make an initial reactor and pick the material to perturb
#Step 2 Solve for the initial keff and adjoint
#Step 3 Confirm with rayleigh quotient nothing is fishy
#Step 4 Create a coopy of the material a reassign the new sigma_a and sigma_r's to the materials affected
#Step 5 Solve for the new k-eff and estimate the perturbed amount from the original matrices
#Step 6 Do this 8 times at 1%, 5%, 10%, 25%, -1%, -5%, -10%, -25% change in material sigma_a
#Step 7 Plot the actual vs predicted