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
        if type(B) not in [list]:
            raise TypeError("B must be a vector")
        
        if type(B).any() not in [int,float]:
            raise TypeError("B components must be a float or integer value")
            
        S = self.__qm.angular_momentum(self.__ss['J'])
        Sx,Sy,Sz = (self.__qm.enlarge_matrix(self.__MI,S['x']),
                         self.__qm.enlarge_matrix(self.__MI,S['y']),
                         self.__qm.enlarge_matrix(self.__MI,S['z']))

        return self.__ub*self.__ss['gS']*(Sx*B[0]+Sy*B[1]+Sz*B[2])

    def nuclear_zeeman(self,B):
        """Calculate the nuclear Zeeman interaction"""
        if type(B) not in [list]:
            raise TypeError("B must be a vector")
        
        if type(B).any() not in [int,float]:
            raise TypeError("B components must be a float or integer value")
            
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
        if type(B) not in [list]:
            raise TypeError("B must be a vector")
        
        if type(B).any() not in [int,float]:
            raise TypeError("B components must be a float or integer value")
            
        return self.hyperfine() + self.electron_zeeman(B) + self.nuclear_zeeman(B)

    def get_field_vector(self,bx,by,bz):
        """Get field vector [bx, by, bz]"""       
        if (type(bx) not in [int,float]) or (type(by) not in [int,float]) or (type(bz) not in [int,float]):
            raise TypeError("B components must be a float or integer value")
            
        return [bx, by, bz]

    def get_field_sweep(self,bmin=0,bmax=1,bnum=1000):
        """Sweep a field along one axis"""
        if (type(bmin) not in [int,float]) or (type(bmax) not in [int,float]):
            raise TypeError("B components must be a float or integer value")
        if type(bnum) not in [int]:
            raise TypeError("B iterator must be an integer value")
            
        return np.around(np.linspace(bmin,bmax,bnum),4) 
    
    def calculate_energy(self,Bz):
        """Calculate the eigenergy"""
        if type(bmin) not in [int,float]:
            raise TypeError("B components must be a float or integer value")
            
        H = [self.get_hamiltonian([0,0,Bz[i]]) for i in range(len(Bz))]
        eigval = [np.linalg.eig(H[i])[0] for i in range(len(H))]

        return [np.sort(np.real(eigval[i]))/self.__h/1e09 for i in range(len(eigval))]

    def clock_transitions_low_field(E,B, show = False):
        transition_counter = 0
        transitions = []
        transition_energies = []
        dE = []
        level0 = []
        level1 = []

        for j in range (len(E)):
            dE.append([])
            level0.append([])
            level1.append([])
            for i in range (len(E)):
                dE[j].append([])
                level0[j].append([])
                level1[j].append([])
                for n in range (len(E[0])):
                    dE[j][i].append([])
                    dE[j][i][n]=(E[i][n]-E[j][n])
                    level0[j][i].append([])
                    level1[j][i].append([])
                    level0[j][i][n],level1[j][i][n]=E[i][n],E[j][n]

        for j in range (len(E)):
            if (j>=len(E)/2):
                Fj=len(E)/4
                mFj =j-(3*len(E)/4-1)
                mIj = mFj-1/2
                mSj = 1/2
            else:    
                Fj=len(E)/4-1    
                mFj = -j+(len(E)/4-1)
                mIj = mFj+1/2
                mSj = -1/2
            if (j==len(E)/2-1):    
                Fj=len(E)/4    
                mFj = -j+(len(E)/4-1)

            #print (mFj)
            for i in range (len(E)):
                if (i>=len(E)/2):
                    Fi=len(E)/4
                    mFi =i-(3*len(E)/4-1)
                    mIi = mFi-1/2
                    mSi = 1/2
                else:
                    Fi=len(E)/4-1
                    mFi = -i+(len(E)/4-1)
                    mIi = mFi+1/2
                    mSi = -1/2
                if (i==len(E)/2-1):
                    Fi=len(E)/4
                    mFi = -i+(len(E)/4-1)

                dmF = mFi-mFj
                dF = Fi-Fj
                #print (mFi,mFj,(i+1)*(1+j))
                if (abs(dmF)==1 and dF==-1):
                    transitions.append([j,i])
                    peak = signal.argrelmin(np.array(dE[i][j]))
                    #print (dE[i][j])
                    transition_energies.append(dE[i][j])
                    if (len(peak[0])!= 0 and show == True):
                        print ("Levels: ("+h_int(mSj)+","+h_int(mIj)+") -> ("+h_int(mSi)+","+h_int(mIi)+")\t Field = "+ str(round(B[peak[0][0]]*1000,2))+" mT\tTransition frequency = "+str(round(dE[i][j][peak[0][0]],6))+" GHz, i,j = %i,%i"%(i,j))
                        print (level1[i][j][peak[0][0]],level0[i][j][peak[0][0]] )
                        plt.plot((B[peak[0][0]], B[peak[0][0]]), (level1[i][j][peak[0][0]],level0[i][j][peak[0][0]]), '#e65c00')

        return(transitions, transition_energies)
