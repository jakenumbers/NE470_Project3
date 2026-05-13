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

##################
## 2G PWR Material
PWR_2G = Material(
    name="PWR, 2G",
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
    description="Two Group PWR Core"
)
    
########################################
#Water Reflector for "Reflector Savings"
WATER_2G = Material(
    name="H$_2$O",
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
    description="Two Group Water Reflector"
)

########################################
# Materials from provided Cross Sections

#############################################
#Water Reflector 17x17 W Assembly next to H2O		##Validated 
H2O_2G = Material(
    name="H$_2$O",
    G=2,
    D=None,				
    sigma_a=[0.001128, 0.009114],	
    sigma_f=[0.0,0.0],			
    nu_sigma_f=[0.0,0.0],		
    chi=[0.0,0.0],			
    sigma_s=[[0.0,0.02229],
            [0.0002373,0.0]],		
    sigma_tr=[0.2342,1.047],		
    Diff_Coeff=None,			
    sigma_r=[0.02342,0.009351],		
    description="Two Group H2O Reflector"
)

#################################################		##Validated 
#Beryllium Reflector 17x17 W Assembly next to H2O    
BE_2G = Material(
    name="Be",
    G=2,
    D=None,					
    sigma_a=[0.0005143, 0.001885],		
    sigma_f=[0.0,0.0],				
    nu_sigma_f=[0.0,0.0],			
    chi=[0.0,0.0],				
    sigma_s=[[0.0,0.009472],			
            [0.0004423,0.0]],
    sigma_tr=[0.5162,1.082],			
    Diff_Coeff=None,						
    sigma_r=[0.009986,0.002327],		
    description="Two Group Beryllium Reflector")

################################################		##Validated 
#Graphite Reflector 17x17 W Assembly next to H2O    
C_2G = Material(
    name="C",
    G=2,
    D=None,					
    sigma_a=[0.0005952, 0.001064],		
    sigma_f=[0.0,0.0],				
    nu_sigma_f=[0.0,0.0],			
    chi=[0.0,0.0],				
    sigma_s=[[0.0,0.004215],			
            [0.0003985,0.0]],			
    sigma_tr=[0.3612,0.4936],			
    Diff_Coeff=None,				
    sigma_r=[0.004810,0.001462],		
    description="Two Group Graphite Reflector"
)

#######################################################		##Validated 
#Stainless Steel Reflector 17x17 W Assembly next to H2O   
SS_2G = Material(
    name="SS",
    G=2,
    D=None,					
    sigma_a=[0.001521, 0.004995],		
    sigma_f=[0.0,0.0],				
    nu_sigma_f=[0.0,0.0],			
    chi=[0.0,0.0],				
    sigma_s=[[0.0,0.0003044],			
            [0.0007263,0.0]],
    sigma_tr=[0.2779,0.2983],			
    Diff_Coeff=None,				
    sigma_r=[0.001825,0.005721],		
    description="Two Group Graphite Reflector"
)


########################################################
# --- 2-GROUP Fuels --- ################################
########################################################

#######################################################################
# Westinghouse 17x17 MOX (wec17) -- 2-group collapsed (k-inf = 1.43686)		##Validated 
MOX_2G = Material(

    ## Macro cross sections 
    name="MOX (W17)",
    G=2,
    D=None,					
    sigma_a=[1.553e-02, 1.262e-01],		
    sigma_f=[7.171e-03, 7.756e-02],				
    nu_sigma_f=[2.065e-02, 1.989e-01],			
    chi=[1.000e+00, 1.902e-09],				
    sigma_s=[[0.0,1.211e-02],			
            [2.205e-03,0.0]],
    sigma_tr=[2.225e-01, 9.178e-01],			
    Diff_Coeff=None,				
    sigma_r=[2.764e-02, 1.284e-01],		
    description="Westinghouse 17x17 MOX assembly, 2-group, k-inf = 1.43686",
)

#######################################################################
# UO2 W17 unrodded 0GWD/MTU	(K-inf = 1.16952)				#Validated
UO2_W17_FRESH_2G = Material(
    ## Macro cross sections 
    name="UO$_2$ 0%BU",
    G=2,
    D=None,					
    sigma_a=[9.506e-03, 9.191e-02],		
    sigma_f=[2.591e-03, 5.435e-02],				
    nu_sigma_f=[6.618e-03, 1.324e-01],			
    chi=[1.000e+00, 2.546e-09],				
    sigma_s=[[0.0,1.698e-02],			
            [1.757e-03,0.0]],
    sigma_tr=[2.285e-01, 8.756e-01],			
    Diff_Coeff=None,				
    sigma_r=[2.641e-02, 9.367e-02],		
    description="0.00 GWD/MTU k-inf = 1.16952",
)

#######################################################################
# UO2 17x17 0.1 GWD/MTU k-inf = 0.8909						#validated
UO2_W17_BU30_2G = Material(
    ## Macro cross sections 
    name="UO$_2$ 30%BU",
    G=2,
    D=None,					
    sigma_a=[1.142e-02, 1.082e-01],		
    sigma_f=[1.850e-03, 4.792e-02],				
    nu_sigma_f=[5.051e-03, 1.308e-01],			
    chi=[1.000e+00, 1.799e-09],				
    sigma_s=[[0.0,1.637e-02],			
            [1.978e-03,0.0]],
    sigma_tr=[2.310e-01, 9.094e-01],			
    Diff_Coeff=None,				
    sigma_r=[2.779e-02, 1.101e-01],		
    description="UO2 17x17 0.1 GWD/MTU k-inf = 0.8909",
)


########################################################
#%% --- 4 Group Materials --- ##########################
########################################################

#############################################
#Water Reflector 17x17 W Assembly next to H2O		##Validated 
H2O_4G = Material(
    name="H$_2$O",
    G=4,
    D=None,
    sigma_a=[0.00031, 0.002475, 0.00145, 0.009114],	
    sigma_f=[0.0,0.0,0.0,0.0],				
    nu_sigma_f=[0.0,0.0,0.0,0.0],			
    chi=[0.0,0.0,0.0,0.0],				
    sigma_s=[[0.0, 0.03058, 0.00006841, 0.000000352],	
             [0.0, 0.0, 0.0742, 0.0003947],		##ERROR: [0.0, 0.0, 0.0742, 0.0003847],
             [0.0, 0.0, 0.0, 0.1018],			
             [0.0, 0.0, 0.0002373, 0.0]],		
    sigma_tr=[0.1805,0.3327,0.3611,1.047],		
    Diff_Coeff=None,					
    sigma_r=[0.03096,0.07707,0.1033,0.009351],		
    description="Four Group H2O Reflector"
)
    
#############################################
## 4G Beryllium reflector 
BE_4G = Material(
    name="Be",
    G=4,
    D=None,
    sigma_a=[-7.871e-04, 1.892e-03, 7.089e-04, 1.885e-03],
    sigma_f=[0.0,0.0,0.0,0.0],				
    nu_sigma_f=[0.0,0.0,0.0,0.0],			
    chi=[1.0, 0.0, 0.0, 0.0],				
    sigma_s=[[0.0, 2.010e-02, 5.926e-11, 3.086e-13],	
             [0.0, 0.0, 2.431e-02, 0.0],		
             [0.0, 0.0, 0.0, 3.482e-02],			
             [0.0, 0.0, 04.423e-04, 0.0]],		
    sigma_tr=[3.332e-01, 7.799e-01, 8.332e-01, 1.082e+00],	
    Diff_Coeff=None,					
    sigma_r=[1.932e-02, 2.621e-02, 3.553e-02, 2.327e-03],	
    description="Beryllium reflector, 4-group, 17x17 W assembly next to H2O."
)
    
##############################################
## 4G Graphite reflector
C_4G = Material(
    name="C",
    G=4,
    D=None,
    sigma_a=[2.231e-04, 1.133e-03, 3.954e-04, 1.064e-03],
    sigma_f=[0.0,0.0,0.0,0.0],				
    nu_sigma_f=[0.0,0.0,0.0,0.0],			
    chi=[1.0, 0.0, 0.0, 0.0],			
    sigma_s=[[0.0, 1.087e-02, 7.451e-11, 2.071e-13],	
             [0.0, 0.0, 1.044e-02, 0.0],		##ERROR: [0.0, 0.0, 0.0742, 0.0003847],
             [0.0, 0.0, 0.0, 1.438e-02],			
             [0.0, 0.0, 3.985e-04, 0.0]],		
    sigma_tr=[2.604e-01, 4.566e-01, 4.601e-01, 4.936e-01],		
    Diff_Coeff=None,					
    sigma_r=[1.110e-02, 1.158e-02, 1.478e-02, 1.462e-03],	
    description="Graphite reflector, 4-group, 17x17 W assembly next to H2O."
)


########################################################
#%% --- 4 Group Fuels --- ##############################
########################################################

##################################
## 4-GROUP MOX			
MOX_4G = Material(
    name="MOX (W17)",
    G=4,
    D=None,
    sigma_a=[5.011e-03, 1.741e-02, 6.722e-02, 1.262e-01],
    sigma_f=[4.100e-03, 6.709e-03, 2.429e-02, 7.756e-02],				
    nu_sigma_f=[1.244e-02, 1.891e-02, 6.742e-02, 1.989e-01],			
    chi=[9.953e-01, 4.683e-03, 3.512e-07, 1.902e-09],			
    sigma_s=[[0.0, 4.121e-02, 9.151e-05, 4.688e-07],	
             [0.0, 0.0, 8.484e-02, 4.490e-04],		
             [0.0, 0.0, 0.0, 9.931e-02],			
             [0.0, 0.0, 2.205e-03, 0.0]],		
    sigma_tr=[1.762e-01, 3.907e-01, 4.688e-01, 9.178e-01],		
    Diff_Coeff=None,					
    sigma_r=[4.631e-02, 1.027e-01, 1.665e-01, 1.284e-01],	
    description="Westinghouse 17x17 MOX assembly, 4-group, k-inf = 1.42758"
)

##################################
## UO2 W17 fesh fuel
UO2_W17_FRESH_4G = Material(
    name="UO$_2$ 0%BU",
    G=4,
    D=None,
    sigma_a=[2.458e-03, 1.013e-02, 3.525e-02, 9.191e-02],
    sigma_f=[1.630e-03, 1.632e-03, 7.784e-03, 5.435e-02],				
    nu_sigma_f=[4.487e-03, 4.487e-03, 1.896e-02, 1.324e-01],			
    chi=[9.955e-01, 4.545e-03, 4.701e-07, 2.546e-09],		
    sigma_s=[[0.0, 4.286e-02, 9.518e-05, 4.882e-07],	
             [0.0, 0.0, 8.870e-02, 4.690e-04],		
             [0.0, 0.0, 0.0, 1.075e-01],			
             [0.0, 0.0, 1.757e-03, 0.0]],		
    sigma_tr=[1.765e-01, 3.839e-01, 4.285e-01, 8.756e-01],		
    Diff_Coeff=None,					
    sigma_r=[4.542e-02, 9.931e-02, 1.427e-01, 9.367e-02],	
    description="UO2 W17x17, unrodded, fresh fuel, k-inf = 1.16926"
)

##################################
## UO2 W17 BU30
UO2_W17_BU30_4G = Material(
    name="UO$_2$ 30%BU",
    G=4,
    D=None,
    sigma_a=[2.350e-03, 9.728e-03, 4.750e-02, 1.082e-01],
    sigma_f=[1.533e-03, 8.379e-04, 4.663e-03, 4.792e-02],				
    nu_sigma_f=[4.305e-03, 2.198e-03, 1.243e-02, 1.308e-01],		
    chi=[9.955e-01, 4.494e-03, 3.321e-07, 1.799e-09],		
    sigma_s=[[0.0, 4.334e-02, 9.623e-05, 4.939e-07],	
             [0.0, 0.0, 8.993e-02, 4.754e-04],		
             [0.0, 0.0, 0.0, 1.023e-01],			
             [0.0, 0.0, 1.978e-03, 0.0]],		
    sigma_tr=[1.765e-01, 3.879e-01, 4.436e-01, 9.094e-01],		
    Diff_Coeff=None,					
    sigma_r=[4.579e-02, 1.001e-02, 1.498e-01, 1.101e-01],	
    description="UO2 W17x17, unrodded, 30.1 GWD/MTU, k-inf = 0.89157"
)

#Rodded lattice at BOC with AIC Control Rods W17x17
Rodded_AIC_4G = Material(
    name="Rodded AIC",
    G=4,
    D=None,
    sigma_a=[0.002662, 0.01248, 0.05131, 0.1198],
    sigma_f=[0.001593,0.001639,0.007843,0.05509],
    nu_sigma_f=[0.004397,0.003987,0.01911,0.1342],
    chi=[0.9954,0.004558,0.0000004861,0.000000002633],
    sigma_s=[[0.0, 0.04249, 0.00009414, 0.0000004835],
             [0.0, 0.0, 0.08442, 0.0004459],
             [0.0, 0.0, 0.0, 0.09742],
             [0.0, 0.0, 0.002183, 0.0]],
    sigma_tr=[0.1799,0.3853,0.442,0.8676],
    Diff_Coeff=None,
    sigma_r=[0.04525,0.09735,0.1487,0.122],
    description="Four Group Rodded Fuel with AIC Control Rods")

#Rodded lattice at BOC with B4C Control Rods W17x17
Rodded_B4C_4G = Material(
    name="Rodded B4C",
    G=4,
    D=None,
    sigma_a=[0.002724, 0.01512, 0.06172, 0.1213],
    sigma_f=[0.001611,0.001632,0.007953,0.05541],
    nu_sigma_f=[0.004441,0.003972,0.01938,0.135],
    chi=[0.9954,0.004562,0.0000004919,0.000000002664],
    sigma_s=[[0.0, 0.04245, 0.00009367, 0.000000481],
             [0.0, 0.0, 0.08367, 0.0004412],
             [0.0, 0.0, 0.0, 0.09683],
             [0.0, 0.0, 0.00216, 0.0]],
    sigma_tr=[0.1792,0.3898,0.4536,0.8739],
    Diff_Coeff=None,
    sigma_r=[0.04527,0.09923,0.1586,0.1234],
    description="Four Group Rodded Fuel with B4C Control Rods")

#Fresh Unrodded Fuel W17x17
FRESH_FUEL_4G = Material(
    name="W17 Fuel 0%BU",
    G=4,
    D=None,
    sigma_a=[0.002458, 0.01013, 0.03525, 0.09191],
    sigma_f=[0.00163,0.001632,0.007784,0.05435],
    nu_sigma_f=[0.004487,0.003971,0.01896,0.1324],
    chi=[0.9955,0.004545,0.0000004701,0.000000002546],
    sigma_s=[[0.0, 0.04286, 0.00009518, 0.0000004882],
             [0.0, 0.0, 0.0887, 0.000469],
             [0.0, 0.0, 0.0, 0.1075],
             [0.0, 0.0, 0.001757, 0.0]],
    sigma_tr=[0.1765,0.3839,0.4285,0.8756],
    Diff_Coeff=None,
    sigma_r=[0.04542,0.09931,0.1427,0.09367],
    description="Four Group Fresh Fuel")

# Typical PWR homogenized core (textbook four-group values used for the base
# homogeneous configuration in the project statement).
PWR_4G = Material(
    name="PWR, 4G",
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
    "PWR":		PWR_2G,
    "WATER":		WATER_2G,
    "H2O": 		H2O_2G,
    "Be":  		BE_2G,
    "C":   		C_2G,
    "SS":  		SS_2G,
    "MOX_wec17_2G": 	MOX_2G,
    "UO2_BU0__2G":	UO2_W17_FRESH_2G,
    "UO2_BU30_2G":	UO2_W17_BU30_2G
}
    

lib_4G = {
    "H2O":       	H2O_4G,
    "BE_4G":		BE_4G,
    "C_4G":		C_4G,
    "PWR_4G":		PWR_4G,
    "MOX_4G":		MOX_4G,
    "UO2_W17_FRESH_4G":	UO2_W17_FRESH_4G,
    "UO2_W17_BU30_4G":	UO2_W17_BU30_4G,
    "Rodded_AIC_4G": Rodded_AIC_4G,
    "Rodded_B4C_4G": Rodded_B4C_4G,
    "FRESH_FUEL_4G": FRESH_FUEL_4G

}


##########################################
## 
def list_materials(groups=2):
    lib = lib_2G if groups == 2 else lib_4G
    return sorted(lib)
    
    
    
    
    
    

