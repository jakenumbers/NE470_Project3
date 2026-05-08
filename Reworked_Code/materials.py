"""
materials.py

Cross-section library for both 2-group and 4-group cross sections

Each :class:`Material` carries:
    G                     : number of energy groups
    D[g]                  : diffusion coefficient                [cm]
    sigma_a[g]            : effective absorption                 [1/cm]
    sigma_f[g]            : fission cross section
    nu_sigma_f[g]         : nu * fission                         [1/cm]
    chi[g]                : fission spectrum (sums to 1)         [-]
    sigma_s[g_from][g_to] : group-to-group scattering            [1/cm]
    sigma_tr[g]           : transport cross section              [1/cm]
    Diff_Coeff[g]         : diffusion coefficient if directly given [cm]
    sigma_r[g]            : removal cross section                [1/cm]
 
"""
########################################################################
## Materials Program ###################################################
########################################################################


import numpy as np


#####################
## Define class #####
#####################


class Material:
    """Class for a single material's group-wise cross sections."""

    def __init__(
        self,
        ## Initialize values
        name,
        G,
        D,
        sigma_a,
        sigma_f,
        nu_sigma_f,
        chi,
        sigma_s,
        sigma_tr=None,
        Diff_Coeff=None,
        sigma_r=None,
        description=""):
        
        #####################       
        self.name = name
        self.G = int(G)
        ## Define diffusion coefficient if not already defined
        self.D = (np.asarray(Diff_Coeff, dtype=float)
                  if Diff_Coeff is not None
                  else np.asarray(1.0 / (3.0 * np.array(sigma_tr)), dtype=float))
                  
        #####################
        self.sigma_a = np.asarray(sigma_a, dtype=float)
        self.nu_sigma_f = np.asarray(nu_sigma_f, dtype=float)
        self.sigma_f = np.asarray(sigma_f, dtype=float)
        self.chi = np.asarray(chi, dtype=float)
        self.sigma_s = np.asarray(sigma_s, dtype=float)  # [g_from, g_to]
        self.sigma_tr = (np.asarray(sigma_tr, dtype=float) if sigma_tr is not None else None)
        self.description = description
        self.sigma_r = np.asarray(sigma_r, dtype=float)
        if sigma_s is None:
            # Assume no skip scattering, no up-scattering
            self.sigma_s = np.zeros((G, G))
            for g in range(G-1):
                self.sigma_s[g,g+1]= sigma_r[g] - sigma_a[g]
        self._validate()

        
###############################################################
#valdiation code to ensure material has all required properties

    def _validate(self):
        for name, arr in (
            ("D", self.D),
            ("sigma_a", self.sigma_a),
            ("nu_sigma_f", self.nu_sigma_f),
            ("chi", self.chi),
            ("sigma_r", self.sigma_r)):
            
            #####################
            if arr.shape != (self.G,):
                raise ValueError(
                    f"Material {self.name!r}: '{name}' has shape {arr.shape}, "
                    f"expected ({self.G},)")
                    
                    
        #####################
        if self.sigma_s.shape != (self.G, self.G):
            raise ValueError(
                f"Material {self.name!r}: sigma_s shape {self.sigma_s.shape}, "
                f"expected ({self.G},{self.G})")


###########################################
# fissile material check function if needed
    def is_fissile(self):
        return float(self.nu_sigma_f.sum()) > 0.0




########################################################
# --- 2-GROUP MATERIALS --- ############################
########################################################



###################################
# Materials for Baseline Validation


## where did these benchmark/baseline values come from? 


PWR_2G = Material(
    name="PWR_Core_2G",
    G=2,
    D=None,
    sigma_a=[0.01207,0.1210], 			
    sigma_f=[0.00332,0.07537],			
    nu_sigma_f=[0.008476,0.18514],	
    chi=[1.0,0.0],
    sigma_s=[[0.0,0.01412],[0.0,0.0]],
    sigma_tr=None,
    Diff_Coeff=[1.2627,0.3543],
    sigma_r=[0.02619,0.1210],
    description="Two Group PWR Core")
    
########################################
#Water Reflector for "Reflector Savings"

WATER_2G = Material(
    name="Water_Reflector_2G",
    G=2,
    D=None,
    sigma_a=[0.0004, 0.0197],
    sigma_f=[0.0,0.0],
    nu_sigma_f=[0.0,0.0],
    chi=[0.0,0.0],
    sigma_s=[[0.0,0.0494],
            [0.0,0.0]],
    sigma_tr=None,
    Diff_Coeff=[1.13,0.16],
    sigma_r=[0.0494,0.0197],
    description="Two Group Water Reflector")

########################################
# Materials from provided Cross Sections


#############################################
#Water Reflector 17x17 W Assembly next to H2O

H2O_2G = Material(
    name="H2O_2G",
    G=2,
    D=None,				##ok
    sigma_a=[0.001128, 0.009114],	##ok
    sigma_f=[0.0,0.0],			##ok
    nu_sigma_f=[0.0,0.0],		##ok
    chi=[0.0,0.0],			##ok
    sigma_s=[[0.0,0.02229],
            [0.0002373,0.0]],		##ok
    sigma_tr=[0.2342,1.047],		##ok
    Diff_Coeff=None,			##ok
    sigma_r=[0.02342,0.009351],		##ok
    description="Two Group H2O Reflector")

#################################################
#Beryllium Reflector 17x17 W Assembly next to H2O    

BE_2G = Material(
    name="BE_2G",
    G=2,
    D=None,					##ok
    sigma_a=[0.0005143, 0.001885],		##ok
    sigma_f=[0.0,0.0],				##ok
    nu_sigma_f=[0.0,0.0],			##ok
    chi=[0.0,0.0],				##ok
    sigma_s=[[0.0,0.009472],			##ok
            [0.0004423,0.0]],
    sigma_tr=[0.5162,1.082],			##ok
    Diff_Coeff=None,				##ok		
    sigma_r=[0.009986,0.002327],		###ERROR: sigma_r=[0.00986,0.002327],	
    description="Two Group Beryllium Reflector")

################################################
#Graphite Reflector 17x17 W Assembly next to H2O    

C_2G = Material(
    name="C_2G",
    G=2,
    D=None,					##ok
    sigma_a=[0.0005952, 0.001064],		##ok
    sigma_f=[0.0,0.0],				##ok
    nu_sigma_f=[0.0,0.0],			##ok
    chi=[0.0,0.0],				##ok
    sigma_s=[[0.0,0.004215],			
            [0.0003985,0.0]],			##ok
    sigma_tr=[0.3612,0.4936],			##ok
    Diff_Coeff=None,				##ok
    sigma_r=[0.004810,0.001462],		##ok
    description="Two Group Graphite Reflector")

#######################################################
#Stainless Steel Reflector 17x17 W Assembly next to H2O   
 
SS_2G = Material(
    name="SS_2G",
    G=2,
    D=None,					##ok
    sigma_a=[0.001521, 0.004995],		##ok
    sigma_f=[0.0,0.0],				##ok
    nu_sigma_f=[0.0,0.0],			##ok
    chi=[0.0,0.0],				##ok
    sigma_s=[[0.0,0.0003044],			##ok
            [0.0007263,0.0]],
    sigma_tr=[0.2779,0.2983],			##ok
    Diff_Coeff=None,				##ok
    sigma_r=[0.001825,0.005721],		##ok
    description="Two Group Graphite Reflector")







########################################################
#%% --- 4 Group Materials --- ##########################
########################################################


## what about the other 4 group reflectors?? Beryllium, and Graphite?


#############################################
#Water Reflector 17x17 W Assembly next to H2O
H2O_4G = Material(
    name="H2O_4G",
    G=4,
    D=None,
    sigma_a=[0.00031, 0.002475, 0.00145, 0.009114],	##ok
    sigma_f=[0.0,0.0,0.0,0.0],				##ok
    nu_sigma_f=[0.0,0.0,0.0,0.0],			##ok
    chi=[0.0,0.0,0.0,0.0],				##ok
    sigma_s=[[0.0, 0.03058, 0.00006841, 0.000000352],	##
             [0.0, 0.0, 0.0742, 0.0003947],		##ERROR: [0.0, 0.0, 0.0742, 0.0003847],
             [0.0, 0.0, 0.0, 0.1018],			##ok
             [0.0, 0.0, 0.0002373, 0.0]],		##ok
    sigma_tr=[0.1805,0.3327,0.3611,1.047],		##ok
    Diff_Coeff=None,					##ok
    sigma_r=[0.03096,0.07707,0.1033,0.009351],		##ok
    description="Four Group H2O Reflector")

# Typical PWR homogenized core (textbook four-group values used for the base
# homogeneous configuration in the project statement).
PWR_4G = Material(
    name="PWR, homogeneous, 4G",
    G=4,
    D=[2.1623, 1.0867, 0.6318, 0.3543],
    sigma_a=[0.004946, 0.002840, 0.03053, 0.1210],
    sigma_f=[0.003378, 0.0004850, 0.006970, 0.07527],
    nu_sigma_f=[0.009572, 0.001193, 0.01768, 0.18514],
    chi=[1.0, 0.0, 0.0, 0.0],
    sigma_s=None,
    sigma_tr=None,
    Diff_Coeff=[2.1623, 1.0867, 0.6318, 0.3543],
    sigma_r=[0.08795, 0.06124, 0.09506, 0.1210],
    description="Four-group homogenized PWR core (Lamarsh-style typical PWR).",
)





##########################
# Material Library look-up
lib_2G = {
    "PWR": PWR_2G,
    "WATER": WATER_2G,
    "H2O": H2O_2G,
    "Be":  BE_2G,
    "C":   C_2G,
    "SS":  SS_2G}

lib_4G = {
    "H2O":        H2O_4G,
    "PWR":        PWR_4G,
}


##########################################
## 
def list_materials(groups=2):
    lib = lib_2G if groups == 2 else lib_4G
    return sorted(lib)
    
    
    
    
    
    

