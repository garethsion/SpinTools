import numpy as np

class Qmech:
    def __init__(self):
        return

    def angular_momentum(self,J):
        """ Find the angular momentum operator for a given total angular momentum J=L+S"""

        if type(J) not in [int,float]:
            raise TypeError("J must be a non-negative, real, numerical value")

        alpha = lambda mz,J : (J*(J+1)-mz*(mz+1))**0.5

        dim=int(2*J+1)

        J_plus = np.zeros((dim,dim))
        J_minus = np.zeros((dim,dim))
        Jz = np.zeros((dim,dim))

        for n in range(dim):
            for m in range(dim):
                if n==m+1:
                    J_plus[n,m] = alpha(J-n,J)
                if n==m-1:
                    J_minus[n,m] = alpha(J-m,J)
                if n==m:
                    Jz[n,m] = J-n

        Jx = 0.5*(J_plus+J_minus)
        Jy = -0.5*1j*(J_plus-J_minus)

        dic = {'x':Jx, 'y':Jy, 'z':Jz }
        return dic

    def enlarge_matrix(self,M,J):
        """ Enlarge a matrix by taking the Kronecker product"""
        return np.kron(np.identity(M),J)
    
    def Pauli_matrix(self,M=1,op='x'):
        return np.kron(np.identity(M),self.angular_momentum(.5)[op])