"""
criticality_search.py
=====================

Search for the critical dimension of a homogeneous (or partially layered)
reactor that satisfies k_eff = 1.

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

from geometry import Region, Regions, Mesh
from matrix_builder import build_matrices
from solver import solve_keff

#%% Criticality Search Function
def crit_search(core_mat, nodes_density: float, width_low: float, width_high: float, progressprint:bool,
                target_k: float = 1.0, tol_k: float = 1e-5, max_iter: int = 100):
    """Iterates through different core widths to find desired k-eff
        only works with bare core"""
    
    #Solve for lowest width and highest width bare cores. If both are below or above k exit and return error
    wl_num_nodes = int(width_low * nodes_density)
    regs = Regions(Region("core", w=width_low, n=wl_num_nodes, mat=core_mat))
    mesh = Mesh(regs)
    A, F = build_matrices(mesh)
    wl_res = solve_keff(A, F)
    
    wh_num_nodes = int(width_high * nodes_density)
    regs = Regions(Region("core", w=width_high, n=wh_num_nodes, mat=core_mat))
    mesh = Mesh(regs)
    A, F = build_matrices(mesh)
    wh_res = solve_keff(A, F)
    
    if wl_res['k'] < 1.0 and wh_res['k'] < 1.0 or wl_res['k'] > 1.0 and wh_res['k'] > 1.0:
        print(wl_res['k'],wh_res['k'])
        raise Exception('Minimum and Maximum Core Widths are both below or above k-eff = 1.0')
    
    width = (width_high + width_low)/2
    wl = width_low
    wh = width_high
    for it in range(max_iter):
        num_nodes = int(width * nodes_density)
        regs = Regions(Region("core",w=width,n=num_nodes,mat=core_mat))
        mesh = Mesh(regs)
        A, F = build_matrices(mesh)
        wm_res = solve_keff(A,F)
        if progressprint == True:
            print(f"Current k-eff: {wm_res['k']:.5f}   Current Width: {width:.3f}")
        if wm_res['k'] > 1.0:
            wh = width 
            width = (wl + width)/2
        elif wm_res['k'] <1.0:
            wl = width
            width = (wh + width)/2
        
        if abs(wm_res['k'] - target_k) < tol_k:
            return wm_res, width, mesh
        
#%% Reflector Savings
def reflector_savings(core_mat, reflector_mat, reflector_width, nodes_density: float, width_low: float, width_high: float,
                 progressprint:bool, target_k: float = 1.0, tol_k: float = 1e-5, max_iter: int = 100):
    """Iterates through different core widths to find desired k-eff
        only works with bare core"""
    
    #Solve for lowest width and highest width bare cores. If both are below or above k exit and return error
    ref_nodes = int(reflector_width * nodes_density)
    wl_num_nodes = int(width_low * nodes_density)
    regs = Regions(Region("LR", w=reflector_width, n=ref_nodes, mat=reflector_mat),
                   Region("core", w=width_low, n=wl_num_nodes, mat=core_mat),
                   Region("RR", w=reflector_width, n=ref_nodes, mat=reflector_mat))
    mesh = Mesh(regs)
    A, F = build_matrices(mesh)
    wl_res = solve_keff(A, F)
    
    wh_num_nodes = int((width_high)* nodes_density)
    regs = Regions(Region("LR", w=reflector_width, n=ref_nodes, mat=reflector_mat),
                   Region("core", w=width_high, n=wh_num_nodes, mat=core_mat),
                   Region("RR", w=reflector_width, n=ref_nodes, mat=reflector_mat))
    mesh = Mesh(regs)
    A, F = build_matrices(mesh)
    wh_res = solve_keff(A, F)
    
    if wl_res['k'] < 1.0 and wh_res['k'] < 1.0 or wl_res['k'] > 1.0 and wh_res['k'] > 1.0:
        print(wl_res['k'],wh_res['k'])
        raise Exception('Minimum and Maximum Core Widths are both below or above k-eff = 1.0')
    
    width = (width_high + width_low)/2
    wl = width_low
    wh = width_high
    for it in range(max_iter):
        num_nodes = int(width * nodes_density)
        regs = Regions(Region("LR", w=reflector_width, n=ref_nodes, mat=reflector_mat),
                       Region("core", w=width, n=num_nodes, mat=core_mat),
                       Region("RR", w=reflector_width, n=ref_nodes, mat=reflector_mat))
        mesh = Mesh(regs)
        A, F = build_matrices(mesh)
        wm_res = solve_keff(A,F)
        if progressprint == True:
            print(f"Current k-eff: {wm_res['k']:.5f}   Current Width: {width:.3f}")
        if wm_res['k'] > 1.0:
            wh = width 
            width = (wl + width)/2
        elif wm_res['k'] <1.0:
            wl = width
            width = (wh + width)/2
        
        if abs(wm_res['k'] - target_k) < tol_k:
            return wm_res, width, mesh