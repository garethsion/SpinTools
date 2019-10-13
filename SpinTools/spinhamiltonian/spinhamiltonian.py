from scipy.constants import physical_constants as spc
from SpinTools.qmech import qmech as qm
from SpinTools.spinhamiltonian import spinsystems as ss
import numpy as np

class SpinHamiltonian:
    def __init__(self,species):
        self.__ub = spc["Bohr magneton"][0]
        self.__un = spc["nuclear magneton"][0]
        self.__h = spc["Planck constant"][0]
        
        spinsystem = ss.SpinSystems()
        self.__ss = spinsystem.systems(species)
        
        self.__MJ = int((self.__ss.J*2)+1)
        self.__MI = int((self.__ss.I*2)+1)
        
        self.__qm = qm.Qmech()
        return
    
    def electron_zeeman(self,B):
        """Electron Zeeman interaction"""
        S = self.__qm.angular_momentum(self.__ss.J)
        Sx,Sy,Sz = (self.__qm.enlarge_matrix(self.__MI,S['x']),
                         self.__qm.enlarge_matrix(self.__MI,S['y']),
                         self.__qm.enlarge_matrix(self.__MI,S['z']))

        return self.__ub*self.__ss.gS*(Sx*B[0]+Sy*B[1]+Sz*B[2])

    def nuclear_zeeman(self,B):
        """Nuclear Zeeman interaction"""
        I = self.__qm.angular_momentum(self.__ss.I)
        Ix,Iy,Iz = (self.__qm.enlarge_matrix(self.__MJ,I['x']),
                         self.__qm.enlarge_matrix(self.__MJ,I['y']),
                         self.__qm.enlarge_matrix(self.__MJ,I['z']))
        return self.__un*self.__ss.gI*((Ix*B[0]+Iy*B[1]+Iz*B[2]))
    
    def hyperfine(self):
        """Hyperfine interaction"""
        S = self.__qm.angular_momentum(self.__ss.J)
        I = self.__qm.angular_momentum(self.__ss.I)
        return self.__h * self.__ss.A * (np.kron(S['x'],I['x']) + 
                np.kron(S['y'],I['y']) + np.kron(S['z'],I['z']))

    def get_hamiltonian(self,B):
        """Get the full spin hamiltonian"""
        return self.hyperfine() + self.electron_zeeman(B) + self.nuclear_zeeman(B)

    def get_field_vector(self,bx,by,bz):
        """Get field vector [bx, by, bz]"""
        return [bx, by, bz]

    def get_field_sweep(self,bmin=0,bmax=1,bnum=1000):
        """Sweep a field along one axis"""
        return np.around(np.linspace(bmin,bmax,bnum),4) 
    
    def calculate_energy(self,Bz):
        """Calculate the eigenergy"""
        H = [self.get_hamiltonian([0,0,Bz[i]]) for i in range(len(Bz))]
        eigval = [np.linalg.eig(H[i])[0] for i in range(len(H))]
        
#         Bz = self.get_field_sweep(bmin=b_range[0],bmax=b_range[1],bnum=b_range[2])
#         Bvectors = [self.get_field_vector(0,0,Bz[i]) for i in range(len(Bz))]
#         H = [self.get_hamiltonian(Bvectors[i]) for i in range(len(Bvectors))]

        return [np.sort(np.real(eigval[i]))/self.__h/1e09 for i in range(len(eigval))]
