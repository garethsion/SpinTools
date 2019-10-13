from scipy.constants import physical_constants as spc
from SpinTools.qmech import qmech as qm
from SpinTools.logger import logger
import numpy as np
import json

class SpinHamiltonian:
    def __init__(self,species):
        """ Constructor method """
        if type(species) not in [str]:
            raise TypeError("Species must be a string")

        self.__ub = spc["Bohr magneton"][0]
        self.__un = spc["nuclear magneton"][0]
        self.__h = spc["Planck constant"][0]
        
        try:
            with open('../../species.json', 'r') as f:
                species_string = f.read()    
            sp = json.loads(species_string)
        except FileNotFoundError:
            "The species json file is not present in the package directory"
        
        self.__ss = sp[species]
        
        self.__MJ = int((self.__ss['J']*2)+1)
        self.__MI = int((self.__ss['I']*2)+1)
        
        self.__qm = qm.Qmech()

        # self.__logger = logger.Logger().logger('../../spinhamiltonianlog')
        # print(self.__logger)
        return
    
    def electron_zeeman(self,B):
        """Calculate the electron Zeeman interaction"""
        S = self.__qm.angular_momentum(self.__ss['J'])
        Sx,Sy,Sz = (self.__qm.enlarge_matrix(self.__MI,S['x']),
                         self.__qm.enlarge_matrix(self.__MI,S['y']),
                         self.__qm.enlarge_matrix(self.__MI,S['z']))

        return self.__ub*self.__ss['gS']*(Sx*B[0]+Sy*B[1]+Sz*B[2])

    def nuclear_zeeman(self,B):
        """Calculate the nuclear Zeeman interaction"""
        I = self.__qm.angular_momentum(self.__ss['I'])
        Ix,Iy,Iz = (self.__qm.enlarge_matrix(self.__MJ,I['x']),
                         self.__qm.enlarge_matrix(self.__MJ,I['y']),
                         self.__qm.enlarge_matrix(self.__MJ,I['z']))
        return self.__un*self.__ss['gI']*((Ix*B[0]+Iy*B[1]+Iz*B[2]))
    
    def hyperfine(self):
        """Calculate the hyperfine interaction"""
        S = self.__qm.angular_momentum(self.__ss['J'])
        I = self.__qm.angular_momentum(self.__ss['I'])
        return self.__h * self.__ss['A'] * (np.kron(S['x'],I['x']) + 
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

        return [np.sort(np.real(eigval[i]))/self.__h/1e09 for i in range(len(eigval))]

    def population_probability(B_sweep, E, B0,Tmin=0,Tmax=0.2):
        """Calculate the probabilities for an electron to be in each level """
        Temp = np.linspace(Tmin,Tmax,1001)
        P = []
        for n in range (MI*2):
            P.append([])
        levels = np.linspace(0,2*MI-1,2*MI, dtype = int)
        for T in Temp:
            En, Zn = get_E(B_sweep, E, B0,T, False)
            Z = np.sum(Zn)
            for level in levels:
                P[level].append(Zn[level]/Z)
    
        # print (np.shape(P))
        # sns.set_style('white')
        
        # colour1 = Color("darkslateblue")
        # colour2 = Color("orange")
        # colours = list(colour2.range_to(colour1,MI*2))
        
        # for level in levels:
        #     plt.plot(np.multiply(Temp,1000),P[level], label = level, color = str(colours[level]))
            
        # plt.title('Probabilities for electron to be in each level (GND = 0), %.1fmT'%(B0*1000))
        # plt.xlabel('Temperature (mK)')
        # plt.ylabel('P')
        # #plt.legend()
        # plt.show()
        return P
