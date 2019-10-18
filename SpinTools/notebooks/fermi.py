from SpinTools.spinhamiltonian import spinhamiltonian as sh
import numpy as np
from scipy.constants import physical_constants as spc
h = spc["Planck constant"][0]

ham = sh.SpinHamiltonian("P")

# Bz = ham.get_field_sweep(bmin=0,bmax=1,bnum=10)
# Bz2 = np.array([0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 
# 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9])
Bz2 = np.array([1])
energy = ham.calculate_energy(Bz2)

# [energy[i][j] for i in range(10) for j in range(4)]

np.set_printoptions(suppress=False,precision=3)
Es = ham.FGR_transitions(Bz2, B_drive = [0,0,1])