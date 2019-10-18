from scipy.constants import physical_constants as spc
from SpinTools.qmech import qmech as qm
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
        
        return
    
    def electron_zeeman(self,B):
        """Calculate the electron Zeeman interaction"""

        if type(B) not in [list]:
            raise TypeError("B must be a vector")
        
        # if type(B.any()) not in [int,float]:
        #     raise TypeError("B components must be a float or integer value")
            
        S = self.__qm.angular_momentum(self.__ss['J'])
        Sx,Sy,Sz = (self.__qm.enlarge_matrix(self.__MI,S['x']),
                         self.__qm.enlarge_matrix(self.__MI,S['y']),
                         self.__qm.enlarge_matrix(self.__MI,S['z']))

        return self.__ub*self.__ss['gS']*(Sx*B[0]+Sy*B[1]+Sz*B[2])

    def nuclear_zeeman(self,B):
        """Calculate the nuclear Zeeman interaction"""
        if type(B) not in [list]:
        	raise TypeError("B must be a vector")
        
        # if type(B.any()) not in [int,float]:
        #     raise TypeError("B components must be a float or integer value")
        
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
        self.__logger.info('get_hamiltonian({0}'.format(B))

        if type(B) not in [list]:
        	raise TypeError("B must be a vector")
        
        # if type(np.array(B).any()) not in [int,float]:
        #     raise TypeError("B components must be a float or integer value")
        return self.hyperfine() + self.electron_zeeman(B) + self.nuclear_zeeman(B)

    def get_field_vector(self,bx,by,bz):
        """Get field vector [bx, by, bz]"""     

        if (type(bx) not in [int,float]) or (type(by) not in [int,float]) or (type(bz) not in [int,float]):
            raise TypeError("B components must be a float or integer value")
        
        self.__logger.debug("# return field vector")
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

        if type(Bz) not in [np.ndarray,int,float]:
        	raise TypeError("B components must be a float or integer value")
            
        H = [self.get_hamiltonian([0,0,Bz[i]]) for i in range(len(Bz))]
        
        eigval = [np.linalg.eig(H[i])[0] for i in range(len(H))]

        return [np.sort(np.real(eigval[i]))/self.__h/1e09 for i in range(len(eigval))]

    def calculate_eigenvectors(self,Bz):
        """Calculate the eigenergy"""

        if type(Bz) not in [np.ndarray,int,float]:
        	raise TypeError("B components must be a float or integer value")
            
        H = [self.get_hamiltonian([0,0,Bz[i]]) for i in range(len(Bz))]
        
        eigvec = [np.linalg.eig(H[i])[1] for i in range(len(H))]

        return eigvec
        # return [np.sort(np.real(eigvec[i]))/self.__h/1e09 for i in range(len(eigvec))]

    def FGR_transitions(self,Bz,B_drive=[1,0,0]):
        # MI = M(species[0])

        # (Ix,Iy,Iz) = ang_mo_op(species[0])
        # (Ixb,Iyb,Izb) = (enlarge_I(Ix,MI),enlarge_I(Iy,MI),enlarge_I(Iz,MI))

        # (Sx,Sy,Sz) = ang_mo_op(species[1])
        # (Sxb,Syb,Szb) = (enlarge_S(MI,Sx),enlarge_S(MI,Sy),enlarge_S(MI,Sz))

        Es,gammas,nm,Efs,Eis,Ejs = [],[],[], [], [], []

        eigvec = self.calculate_eigenvectors(Bz)

        H_drive = self.electron_zeeman(B_drive)
        length = len(np.squeeze(np.asarray(eigvec[0][0,:])))

        for n in range(length):
            i = np.squeeze(np.asarray(eigvec[0][:,n]))
            for m in range(n+1,len(i)):
                f = np.squeeze(np.asarray(eigvec[0][:,m]))
                gamma = 1e24*np.abs(np.matmul(f,np.squeeze(np.asarray(np.matmul(H_drive,i)))))

                if gamma > 0 :
                    Ef = np.real(eigvec[1][m])/self.__h/1E9
                    Ei = np.real(eigvec[1][n])/self.__h/1E9
                    E=(np.abs(Ef-Ei))
                    # Es.append([E,gamma,Ef,Ei])
                    Es.append([E])
                    gammas.append(gamma)
                    Efs.append(Ef)
                    Eis.append(Ei)
                    Ejs.append([E,gamma,Ef,Ei])
            #print(Ef,Ei)

        import pdb; pdb.set_trace()
        Es = (np.array(Es))
        # #print(Es[:,1].min())
        # self.__logger.debug('# Sort Es')
        # print(Es)
        # return Es
        # Es = Es[Es[:,0].argsort()]
        # self.__logger.debug('# return Es')
        # return(Es[:,0],Es[:,1],Es[:,2],Es[:,3]) 
