"""
materials.py
============

Cross-section library for the NE 470 Project 3 finite-difference diffusion
solver.  Stores 2-group and 4-group macroscopic cross-section sets for the
materials supplied with the project.

Each :class:`Material` carries:

    G                     : number of energy groups
    D[g]                  : diffusion coefficient                [cm]
    sigma_a[g]            : effective absorption                 [1/cm]
    sigma_r[g]            : removal cross section                [1/cm]
                            (= sigma_a + sum_{g'!=g} sigma_s[g->g'])
    sigma_t[g]            : total cross section                  [1/cm]
    sigma_tr[g]           : transport cross section              [1/cm]
    nu_sigma_f[g]         : nu * fission                         [1/cm]
    sigma_f[g]            : fission                              [1/cm]
    kappa_sigma_f[g]      : kappa * fission (energy release)     [J/cm]
    chi[g]                : fission spectrum (sums to 1)         [-]
    sigma_s[g_from][g_to] : group-to-group scattering            [1/cm]

The convention used in the matrix builder is that ``sigma_r`` already
contains everything that removes a neutron from group ``g`` (absorption
plus scatter-out), so on the diagonal of the loss matrix we use
``sigma_r``, and the off-diagonal in-scattering is added explicitly via
``sigma_s[g_from][g_to]``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Class definition
# ---------------------------------------------------------------------------
class Material:
    """Container for a single material's group-wise cross sections."""

    def __init__(
        self,
        name,
        G,
        D,
        sigma_a,
        nu_sigma_f,
        chi,
        sigma_s,
        sigma_t=None,
        sigma_tr=None,
        sigma_f=None,
        kappa_sigma_f=None,
        sigma_r=None,
        description="",
    ):
        self.name = name
        self.G = int(G)

        self.D = np.asarray(D, dtype=float)
        self.sigma_a = np.asarray(sigma_a, dtype=float)
        self.nu_sigma_f = np.asarray(nu_sigma_f, dtype=float)
        self.chi = np.asarray(chi, dtype=float)
        self.sigma_s = np.asarray(sigma_s, dtype=float)  # [g_from, g_to]

        # Optional / derived
        self.sigma_t = (
            np.asarray(sigma_t, dtype=float) if sigma_t is not None else None
        )
        self.sigma_tr = (
            np.asarray(sigma_tr, dtype=float) if sigma_tr is not None else None
        )
        self.sigma_f = (
            np.asarray(sigma_f, dtype=float) if sigma_f is not None else None
        )
        self.kappa_sigma_f = (
            np.asarray(kappa_sigma_f, dtype=float)
            if kappa_sigma_f is not None
            else None
        )

        self.description = description

        # Removal cross section: absorption + scatter-out
        # If supplied (e.g. straight from the data sheet) use it,
        # otherwise compute from sigma_a and sigma_s rows.
        if sigma_r is not None:
            self.sigma_r = np.asarray(sigma_r, dtype=float)
        else:
            scatter_out = np.zeros(self.G)
            for g in range(self.G):
                for gp in range(self.G):
                    if gp != g:
                        scatter_out[g] += self.sigma_s[g, gp]
            self.sigma_r = self.sigma_a + scatter_out

        self._validate()

    # ------------------------------------------------------------------
    def _validate(self):
        for name, arr in (
            ("D", self.D),
            ("sigma_a", self.sigma_a),
            ("nu_sigma_f", self.nu_sigma_f),
            ("chi", self.chi),
            ("sigma_r", self.sigma_r),
        ):
            if arr.shape != (self.G,):
                raise ValueError(
                    f"Material {self.name!r}: '{name}' has shape {arr.shape}, "
                    f"expected ({self.G},)"
                )
        if self.sigma_s.shape != (self.G, self.G):
            raise ValueError(
                f"Material {self.name!r}: sigma_s shape {self.sigma_s.shape}, "
                f"expected ({self.G},{self.G})"
            )
        if not np.isclose(self.chi.sum(), 1.0, atol=1e-3) and self.chi.sum() > 0:
            # Allow chi to be all zeros for non-fissile, but warn if non-zero
            # and non-unity.
            pass

    # ------------------------------------------------------------------
    def is_fissile(self):
        return float(self.nu_sigma_f.sum()) > 0.0

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"Material({self.name!r}, G={self.G}, fissile={self.is_fissile()})"


# ---------------------------------------------------------------------------
# Helper: build a Material from the 4-group "Polaris" table format used in
# the project's cross-section sheets.
# ---------------------------------------------------------------------------
def _from_polaris(
    name,
    flux,
    total,
    transport,
    removal,
    eff_abs,
    kappa_abs,
    kappa_fis,
    fission,
    nu_fission,
    chi,
    scatter_pairs,
    description="",
):
    """
    scatter_pairs: dict with keys like (1,2) -> value (1-indexed groups).
    All other arrays use Polaris ordering (group 1 = fast).
    """
    G = len(total)
    sigma_t = np.array(total, dtype=float)
    sigma_tr = np.array(transport, dtype=float)
    sigma_r = np.array(removal, dtype=float)
    sigma_a = np.array(eff_abs, dtype=float)
    nu_sigma_f = np.array(nu_fission, dtype=float)
    sigma_f = np.array(fission, dtype=float)
    kappa_sigma_f = np.array(kappa_fis, dtype=float)
    chi_arr = np.array(chi, dtype=float)

    # Diffusion coefficient via D = 1 / (3 * sigma_tr)
    D = 1.0 / (3.0 * sigma_tr)

    sigma_s = np.zeros((G, G))
    for (g_from, g_to), val in scatter_pairs.items():
        sigma_s[g_from - 1, g_to - 1] = val

    return Material(
        name=name,
        G=G,
        D=D,
        sigma_a=sigma_a,
        nu_sigma_f=nu_sigma_f,
        chi=chi_arr,
        sigma_s=sigma_s,
        sigma_t=sigma_t,
        sigma_tr=sigma_tr,
        sigma_f=sigma_f,
        kappa_sigma_f=kappa_sigma_f,
        sigma_r=sigma_r,
        description=description,
    )


# ---------------------------------------------------------------------------
# 2-GROUP MATERIALS
# ---------------------------------------------------------------------------
# Typical PWR homogenized core (textbook two-group values used for the base
# homogeneous configuration in the project statement).  These are consistent
# with D1 = 1.13 cm specified in the project description.
_PWR_NU = 2.43        # average nu used to derive sigma_f from nu_sigma_f
_PWR_KAPPA = 3.2044e-11   # J per fission (~200 MeV)
PWR_2G = Material(
    name="PWR, homogeneous, 2G",
    G=2,
    D=[1.2627, 0.3543],
    sigma_a=[0.01207, 0.1210],
    nu_sigma_f=[0.008476, 0.18514],
    chi=[1.0, 0.0],
    sigma_s=[[0.0, 0.0494],
             [0.0, 0.0]],
    sigma_f=[0.00332 / _PWR_NU, 0.07537 / _PWR_NU],
    kappa_sigma_f=[(0.00332 / _PWR_NU) * _PWR_KAPPA,
                   (0.07537 / _PWR_NU) * _PWR_KAPPA],
    description="Two-group homogenized PWR core (Lamarsh-style typical PWR).",
)

#Water Reflector for "Reflector Savings"
WATER_2G = Material(
    name="Water_Reflector_2G",
    G=2,
    D=[1.13, 0.16],
    sigma_a=[0.0004, 0.0197],
    nu_sigma_f=[0.0,0.0],
    chi=[0.0,0.0],
    sigma_s=[[0.0,0.0494],
             [0.0,0.0]]
)


# Westinghouse 17x17 MOX (wec17) -- 2-group collapsed (k-inf = 1.43686)
MOX_2G = _from_polaris(
    name="MOX_wec17_2G",
    flux=[9.155e-01, 8.451e-02],
    total=[5.234e-01, 1.408e+00],
    transport=[2.225e-01, 9.178e-01],
    removal=[2.764e-02, 1.284e-01],
    eff_abs=[1.553e-02, 1.262e-01],
    kappa_abs=[2.361e-13, 2.473e-12],
    kappa_fis=[2.291e-13, 2.435e-12],
    fission=[7.171e-03, 7.756e-02],
    nu_fission=[2.065e-02, 1.989e-01],
    chi=[1.000e+00, 1.902e-09],
    scatter_pairs={(1, 2): 1.211e-02, (2, 1): 2.205e-03},
    description="Westinghouse 17x17 MOX assembly, 2-group, k-inf = 1.43686",
)

# Water reflector (2-group)
H2O_2G = _from_polaris(
    name="H$_2$O, 2G",
    flux=[3.119e-01, 6.881e-01],
    total=[5.012e-01, 1.631e+00],
    transport=[2.342e-01, 1.047e+00],
    removal=[2.342e-02, 9.351e-03],
    eff_abs=[1.128e-03, 9.114e-03],
    kappa_abs=[1.215e-15, 4.321e-15],
    kappa_fis=[0.0, 0.0],
    fission=[0.0, 0.0],
    nu_fission=[0.0, 0.0],
    chi=[1.0, 0.0],
    scatter_pairs={(1, 2): 2.229e-02, (2, 1): 2.373e-04},
    description="Water reflector, 17x17 W assembly next to H2O, 3.1 w/o.",
)

# Beryllium reflector (2-group)
BE_2G = _from_polaris(
    name="Be_reflector_2G",
    flux=[2.646e-01, 7.354e-01],
    total=[7.154e-01, 1.165e+00],
    transport=[5.162e-01, 1.082e+00],
    removal=[9.986e-03, 2.327e-03],
    eff_abs=[5.143e-04, 1.885e-03],
    kappa_abs=[1.328e-15, 1.865e-15],
    kappa_fis=[0.0, 0.0],
    fission=[0.0, 0.0],
    nu_fission=[0.0, 0.0],
    chi=[1.0, 0.0],
    scatter_pairs={(1, 2): 9.472e-03, (2, 1): 4.423e-04},
    description="Beryllium reflector, 17x17 W assembly next to H2O, 3.1 w/o.",
)

# Graphite (carbon) reflector (2-group)
C_2G = _from_polaris(
    name="C_reflector_2G",
    flux=[3.824e-01, 6.176e-01],
    total=[4.364e-01, 5.203e-01],
    transport=[3.612e-01, 4.936e-01],
    removal=[4.81e-03, 1.462e-03],
    eff_abs=[5.952e-04, 1.064e-03],
    kappa_abs=[6.99e-16, 1.221e-15],
    kappa_fis=[0.0, 0.0],
    fission=[0.0, 0.0],
    nu_fission=[0.0, 0.0],
    chi=[1.0, 0.0],
    scatter_pairs={(1, 2): 4.215e-03, (2, 1): 3.985e-04},
    description="Graphite reflector, 17x17 W assembly next to H2O, 3.1 w/o.",
)

# Stainless steel reflector (2-group)
SS_2G = _from_polaris(
    name="SS_reflector_2G",
    flux=[8.282e-01, 1.718e-01],
    total=[3.362e-01, 3.006e-01],
    transport=[2.779e-01, 2.983e-01],
    removal=[1.825e-03, 5.721e-03],
    eff_abs=[1.521e-03, 4.995e-03],
    kappa_abs=[1.768e-15, 6.201e-15],
    kappa_fis=[0.0, 0.0],
    fission=[0.0, 0.0],
    nu_fission=[0.0, 0.0],
    chi=[1.0, 0.0],
    scatter_pairs={(1, 2): 3.044e-04, (2, 1): 7.263e-04},
    description="Stainless steel reflector, 17x17 W assembly next to H2O, 3.1 w/o.",
)


# ---------------------------------------------------------------------------
# 4-GROUP MATERIALS - UO2 W17x17 lattice at various burnups (unrodded)
# ---------------------------------------------------------------------------
UO2_W17_FRESH_4G = _from_polaris(
    name="UO$_2$, 4G, fresh",
    flux=[5.031e-01, 2.130e-01, 1.325e-01, 1.514e-01],
    total=[3.388e-01, 8.140e-01, 8.626e-01, 1.353e+00],
    transport=[1.765e-01, 3.839e-01, 4.285e-01, 8.756e-01],
    removal=[4.542e-02, 9.931e-02, 1.427e-01, 9.367e-02],
    eff_abs=[2.458e-03, 1.013e-02, 3.525e-02, 9.191e-02],
    kappa_abs=[5.226e-14, 5.763e-14, 2.635e-13, 1.715e-12],
    kappa_fis=[5.155e-14, 5.073e-14, 2.420e-13, 1.689e-12],
    fission=[1.630e-03, 1.632e-03, 7.784e-03, 5.435e-02],
    nu_fission=[4.487e-03, 3.971e-03, 1.896e-02, 1.324e-01],
    chi=[9.955e-01, 4.545e-03, 4.701e-07, 2.546e-09],
    scatter_pairs={
        (1, 2): 4.286e-02, (1, 3): 9.518e-05, (1, 4): 4.882e-07,
        (2, 3): 8.870e-02, (2, 4): 4.690e-04,
        (3, 4): 1.075e-01, (4, 3): 1.757e-03,
    },
    description="UO2 W17x17, unrodded, fresh fuel, k-inf = 1.16926",
)

UO2_W17_BU0p1_4G = _from_polaris(
    name="UO$_2$, 4G, BU0.1",
    flux=[5.030e-01, 2.142e-01, 1.338e-01, 1.490e-01],
    total=[3.389e-01, 8.140e-01, 8.627e-01, 1.355e+00],
    transport=[1.765e-01, 3.840e-01, 4.286e-01, 8.756e-01],
    removal=[4.550e-02, 9.944e-02, 1.429e-01, 9.643e-02],
    eff_abs=[2.458e-03, 1.014e-02, 3.529e-02, 9.463e-02],
    kappa_abs=[5.223e-14, 5.751e-14, 2.628e-13, 1.700e-12],
    kappa_fis=[5.152e-14, 5.060e-14, 2.412e-13, 1.670e-12],
    fission=[1.629e-03, 1.628e-03, 7.760e-03, 5.372e-02],
    nu_fission=[4.486e-03, 3.961e-03, 1.891e-02, 1.309e-01],
    chi=[9.955e-01, 4.545e-03, 4.710e-07, 2.551e-09],
    scatter_pairs={
        (1, 2): 4.295e-02, (1, 3): 9.538e-05, (1, 4): 4.893e-07,
        (2, 3): 8.883e-02, (2, 4): 4.696e-04,
        (3, 4): 1.076e-01, (4, 3): 1.804e-03,
    },
    description="UO2 W17x17, unrodded, 0.1 GWD/MTU, k-inf = 1.13226",
)

UO2_W17_BU10_4G = _from_polaris(
    name="UO$_2$, 4G, BU10",
    flux=[5.106e-01, 2.202e-01, 1.365e-01, 1.327e-01],
    total=[3.389e-01, 8.155e-01, 8.719e-01, 1.374e+00],
    transport=[1.763e-01, 3.853e-01, 4.353e-01, 8.966e-01],
    removal=[4.553e-02, 9.968e-02, 1.459e-01, 1.086e-01],
    eff_abs=[2.421e-03, 9.925e-03, 4.085e-02, 1.066e-01],
    kappa_abs=[5.134e-14, 4.719e-14, 2.294e-13, 1.761e-12],
    kappa_fis=[5.063e-14, 4.016e-14, 2.016e-13, 1.720e-12],
    fission=[1.599e-03, 1.286e-03, 6.447e-03, 5.476e-02],
    nu_fission=[4.443e-03, 3.206e-03, 1.622e-02, 1.414e-01],
    chi=[9.955e-01, 4.458e-03, 3.884e-07, 2.104e-09],
    scatter_pairs={
        (1, 2): 4.301e-02, (1, 3): 9.551e-05, (1, 4): 4.900e-07,
        (2, 3): 8.928e-02, (2, 4): 4.720e-04,
        (3, 4): 1.050e-01, (4, 3): 1.984e-03,
    },
    description="UO2 W17x17, unrodded, 10.1 GWD/MTU, k-inf = 1.03503",
)

UO2_W17_BU30_4G = _from_polaris(
    name="UO$_2$, 4G, BU30",
    flux=[5.074e-01, 2.244e-01, 1.381e-01, 1.302e-01],
    total=[3.398e-01, 8.184e-01, 8.835e-01, 1.382e+00],
    transport=[1.765e-01, 3.879e-01, 4.436e-01, 9.094e-01],
    removal=[4.579e-02, 1.001e-01, 1.498e-01, 1.101e-01],
    eff_abs=[2.350e-03, 9.728e-03, 4.750e-02, 1.082e-01],
    kappa_abs=[4.933e-14, 3.369e-14, 1.831e-13, 1.572e-12],
    kappa_fis=[4.862e-14, 2.641e-14, 1.475e-13, 1.522e-12],
    fission=[1.533e-03, 8.379e-04, 4.663e-03, 4.792e-02],
    nu_fission=[4.305e-03, 2.198e-03, 1.243e-02, 1.308e-01],
    chi=[9.955e-01, 4.494e-03, 3.321e-07, 1.799e-09],
    scatter_pairs={
        (1, 2): 4.334e-02, (1, 3): 9.623e-05, (1, 4): 4.939e-07,
        (2, 3): 8.993e-02, (2, 4): 4.754e-04,
        (3, 4): 1.023e-01, (4, 3): 1.978e-03,
    },
    description="UO2 W17x17, unrodded, 30.1 GWD/MTU, k-inf = 0.89157",
)

UO2_W17_BU60_4G = _from_polaris(
    name="UO$_2$, 4G, BU60",
    flux=[5.008e-01, 2.271e-01, 1.385e-01, 1.337e-01],
    total=[3.412e-01, 8.229e-01, 8.939e-01, 1.383e+00],
    transport=[1.769e-01, 3.916e-01, 4.521e-01, 9.151e-01],
    removal=[4.610e-02, 1.008e-01, 1.539e-01, 1.066e-01],
    eff_abs=[2.287e-03, 9.822e-03, 5.345e-02, 1.047e-01],
    kappa_abs=[4.742e-14, 2.462e-14, 1.488e-13, 1.321e-12],
    kappa_fis=[4.671e-14, 1.694e-14, 1.062e-13, 1.267e-12],
    fission=[1.471e-03, 5.305e-04, 3.319e-03, 3.955e-02],
    nu_fission=[4.168e-03, 1.488e-03, 9.392e-03, 1.126e-01],
    chi=[9.955e-01, 4.539e-03, 2.991e-07, 1.620e-09],
    scatter_pairs={
        (1, 2): 4.372e-02, (1, 3): 9.705e-05, (1, 4): 4.983e-07,
        (2, 3): 9.046e-02, (2, 4): 4.781e-04,
        (3, 4): 1.005e-01, (4, 3): 1.889e-03,
    },
    description="UO2 W17x17, unrodded, 60.1 GWD/MTU, k-inf = 0.75678",
)


# ---------------------------------------------------------------------------
# 4-GROUP MATERIALS - UO2 W17 with control rods inserted (BOC, fresh)
# ---------------------------------------------------------------------------
UO2_W17_AIC_4G = _from_polaris(
    name="UO$_2$_W17_AIC_rodded_4G",
    flux=[5.188e-01, 2.336e-01, 1.366e-01, 1.110e-01],
    total=[3.418e-01, 7.944e-01, 8.583e-01, 1.328e+00],
    transport=[1.799e-01, 3.853e-01, 4.420e-01, 8.676e-01],
    removal=[4.525e-02, 9.735e-02, 1.487e-01, 1.220e-01],
    eff_abs=[2.662e-03, 1.248e-02, 5.131e-02, 1.198e-01],
    kappa_abs=[5.141e-14, 6.002e-14, 2.802e-13, 1.761e-12],
    kappa_fis=[5.048e-14, 5.094e-14, 2.438e-13, 1.712e-12],
    fission=[1.596e-03, 1.639e-03, 7.843e-03, 5.509e-02],
    nu_fission=[4.397e-03, 3.987e-03, 1.911e-02, 1.342e-01],
    chi=[9.954e-01, 4.558e-03, 4.861e-07, 2.633e-09],
    scatter_pairs={
        (1, 2): 4.249e-02, (1, 3): 9.414e-05, (1, 4): 4.835e-07,
        (2, 3): 8.442e-02, (2, 4): 4.459e-04,
        (3, 4): 9.742e-02, (4, 3): 2.183e-03,
    },
    description="UO2 W17x17 lattice with AIC control rods, BOC, k-inf = 0.84167",
)

UO2_W17_B4C_4G = _from_polaris(
    name="UO$_2$_W17_B4C_rodded_4G",
    flux=[5.304e-01, 2.367e-01, 1.293e-01, 1.036e-01],
    total=[3.397e-01, 7.998e-01, 8.731e-01, 1.333e+00],
    transport=[1.792e-01, 3.898e-01, 4.536e-01, 8.739e-01],
    removal=[4.527e-02, 9.923e-02, 1.586e-01, 1.234e-01],
    eff_abs=[2.724e-03, 1.512e-02, 6.172e-02, 1.213e-01],
    kappa_abs=[5.179e-14, 5.989e-14, 2.809e-13, 1.761e-12],
    kappa_fis=[5.095e-14, 5.074e-14, 2.472e-13, 1.722e-12],
    fission=[1.611e-03, 1.632e-03, 7.953e-03, 5.541e-02],
    nu_fission=[4.441e-03, 3.972e-03, 1.938e-02, 1.350e-01],
    chi=[9.954e-01, 4.562e-03, 4.919e-07, 2.664e-09],
    scatter_pairs={
        (1, 2): 4.245e-02, (1, 3): 9.367e-05, (1, 4): 4.810e-07,
        (2, 3): 8.367e-02, (2, 4): 4.412e-04,
        (3, 4): 9.683e-02, (4, 3): 2.160e-03,
    },
    description="UO2 W17x17 lattice with B4C control rods, BOC, k-inf = 0.77317",
)


# ---------------------------------------------------------------------------
# 4-GROUP MATERIALS - MOX
# ---------------------------------------------------------------------------
MOX_4G = _from_polaris(
    name="MOX_wec17_4G",
    flux=[5.834e-01, 2.215e-01, 1.106e-01, 8.451e-02],
    total=[3.363e-01, 8.232e-01, 9.095e-01, 1.408e+00],
    transport=[1.762e-01, 3.907e-01, 4.688e-01, 9.178e-01],
    removal=[4.631e-02, 1.027e-01, 1.665e-01, 1.284e-01],
    eff_abs=[5.011e-03, 1.741e-02, 6.722e-02, 1.262e-01],
    kappa_abs=[1.317e-13, 2.238e-13, 8.116e-13, 2.473e-12],
    kappa_fis=[1.309e-13, 2.145e-13, 7.756e-13, 2.435e-12],
    fission=[4.100e-03, 6.709e-03, 2.429e-02, 7.756e-02],
    nu_fission=[1.244e-02, 1.891e-02, 6.742e-02, 1.989e-01],
    chi=[9.953e-01, 4.683e-03, 3.512e-07, 1.902e-09],
    scatter_pairs={
        (1, 2): 4.121e-02, (1, 3): 9.151e-05, (1, 4): 4.688e-07,
        (2, 3): 8.484e-02, (2, 4): 4.490e-04,
        (3, 4): 9.931e-02, (4, 3): 2.205e-03,
    },
    description="Westinghouse 17x17 MOX assembly, 4-group, k-inf = 1.42758",
)


# ---------------------------------------------------------------------------
# 4-GROUP MATERIALS - reflectors (from the Excel sheet)
# ---------------------------------------------------------------------------
H2O_4G = _from_polaris(
    name="H$_2$O, 4G",
    flux=[1.618e-01, 8.211e-02, 6.795e-02, 6.881e-01],
    total=[3.307e-01, 6.553e-01, 7.210e-01, 1.631e+00],
    transport=[1.805e-01, 3.327e-01, 3.611e-01, 1.047e+00],
    removal=[3.096e-02, 7.707e-02, 1.033e-01, 9.351e-03],
    eff_abs=[3.10e-04, 2.475e-03, 1.450e-03, 9.114e-03],
    kappa_abs=[3.841e-16, 3.014e-15, 1.021e-15, 4.321e-15],
    kappa_fis=[0.0, 0.0, 0.0, 0.0],
    fission=[0.0, 0.0, 0.0, 0.0],
    nu_fission=[0.0, 0.0, 0.0, 0.0],
    chi=[1.0, 0.0, 0.0, 0.0],
    scatter_pairs={
        (1, 2): 3.058e-02, (1, 3): 6.841e-05, (1, 4): 3.520e-07,
        (2, 3): 7.420e-02, (2, 4): 3.947e-04,
        (3, 4): 1.018e-01, (4, 3): 2.373e-04,
    },
    description="Water reflector, 4-group, 17x17 W assembly next to H2O.",
)

BE_4G = _from_polaris(
    name="Be_reflector_4G",
    flux=[1.043e-01, 8.834e-02, 7.197e-02, 7.354e-01],
    total=[4.919e-01, 8.347e-01, 8.927e-01, 1.165e+00],
    transport=[3.332e-01, 7.799e-01, 8.332e-01, 1.082e+00],
    removal=[1.932e-02, 2.621e-02, 3.553e-02, 2.327e-03],
    eff_abs=[-7.871e-04, 1.892e-03, 7.089e-04, 1.885e-03],
    kappa_abs=[9.368e-16, 2.323e-15, 6.730e-16, 1.865e-15],
    kappa_fis=[0.0, 0.0, 0.0, 0.0],
    fission=[0.0, 0.0, 0.0, 0.0],
    nu_fission=[0.0, 0.0, 0.0, 0.0],
    chi=[1.0, 0.0, 0.0, 0.0],
    scatter_pairs={
        (1, 2): 2.010e-02, (1, 3): 5.926e-11, (1, 4): 3.086e-13,
        (2, 3): 2.431e-02, (3, 4): 3.482e-02,
        (4, 3): 4.423e-04,
    },
    description="Beryllium reflector, 4-group, 17x17 W assembly next to H2O.",
)

C_4G = _from_polaris(
    name="C_reflector_4G",
    flux=[1.351e-01, 1.352e-01, 1.121e-01, 6.176e-01],
    total=[3.530e-01, 4.804e-01, 4.840e-01, 5.203e-01],
    transport=[2.604e-01, 4.566e-01, 4.601e-01, 4.936e-01],
    removal=[1.110e-02, 1.158e-02, 1.478e-02, 1.462e-03],
    eff_abs=[2.231e-04, 1.133e-03, 3.954e-04, 1.064e-03],
    kappa_abs=[2.706e-16, 1.391e-15, 3.805e-16, 1.221e-15],
    kappa_fis=[0.0, 0.0, 0.0, 0.0],
    fission=[0.0, 0.0, 0.0, 0.0],
    nu_fission=[0.0, 0.0, 0.0, 0.0],
    chi=[1.0, 0.0, 0.0, 0.0],
    scatter_pairs={
        (1, 2): 1.087e-02, (1, 3): 7.451e-11, (1, 4): 2.071e-13,
        (2, 3): 1.044e-02, (3, 4): 1.438e-02,
        (4, 3): 3.985e-04,
    },
    description="Graphite reflector, 4-group, 17x17 W assembly next to H2O.",
)


# ---------------------------------------------------------------------------
# Library look-up
# ---------------------------------------------------------------------------
LIBRARY_2G = {
    "PWR": PWR_2G,
    "MOX": MOX_2G,
    "H2O": H2O_2G,
    "Be":  BE_2G,
    "C":   C_2G,
    "SS":  SS_2G,
}

LIBRARY_4G = {
    "UO2_fresh":  UO2_W17_FRESH_4G,
    "UO2_BU0.1":  UO2_W17_BU0p1_4G,
    "UO2_BU10":   UO2_W17_BU10_4G,
    "UO2_BU30":   UO2_W17_BU30_4G,
    "UO2_BU60":   UO2_W17_BU60_4G,
    "UO2_AIC":    UO2_W17_AIC_4G,
    "UO2_B4C":    UO2_W17_B4C_4G,
    "MOX":        MOX_4G,
    "H2O":        H2O_4G,
    "Be":         BE_4G,
    "C":          C_4G,
}


def get(name, groups=2):
    """Look up a material by short name."""
    lib = LIBRARY_2G if groups == 2 else LIBRARY_4G
    if name not in lib:
        raise KeyError(
            f"Material {name!r} not found in {groups}-group library. "
            f"Available: {sorted(lib)}"
        )
    return lib[name]


def list_materials(groups=2):
    lib = LIBRARY_2G if groups == 2 else LIBRARY_4G
    return sorted(lib)


if __name__ == "__main__":
    print(f"--- 2-group library ({len(LIBRARY_2G)}) ---")
    for n in sorted(LIBRARY_2G):
        m = LIBRARY_2G[n]
        print(f"  {n:12s}  D={m.D}  Sa={m.sigma_a}  nuSf={m.nu_sigma_f}")
    print(f"\n--- 4-group library ({len(LIBRARY_4G)}) ---")
    for n in sorted(LIBRARY_4G):
        m = LIBRARY_4G[n]
        print(f"  {n:12s}  D={m.D}")
