import numpy as np
import Logger

class Qmech:
    def __init__(self):
        log = Logger.Logger("../../logs/" + __name__)
        self.__logger = log.get_logger()
        self.__logger.info("Qmech() Constructor Called")
        return

    def angular_momentum(self,J):
        """ Find the angular momentum operator for a given total angular momentum J=L+S"""
        self.__logger.info('angular_momentum({0}'.format(J))

        if type(J) not in [int,float]:
            self.__logger.error("Incorrect type {0} passed to method".format(J))
            raise TypeError("J must be a non-negative, real, numerical value")

        alpha = lambda mz,J : (J*(J+1)-mz*(mz+1))**0.5

        dim=int(2*J+1)

        J_plus = np.zeros((dim,dim))
        J_minus = np.zeros((dim,dim))
        Jz = np.zeros((dim,dim))

        self.__logger.debug("# Calculate J+, J-, and Jz ")
        for n in range(dim):
            for m in range(dim):
                if n==m+1:
                    J_plus[n,m] = alpha(J-n,J)
                if n==m-1:
                    J_minus[n,m] = alpha(J-m,J)
                if n==m:
                    Jz[n,m] = J-n

        self.__logger.debug("# Calculate Jx and Jy ")
        Jx = 0.5*(J_plus+J_minus)
        Jy = -0.5*1j*(J_plus-J_minus)

        dic = {'x':Jx, 'y':Jy, 'z':Jz }
        self.__logger.debug("# Return dictionary of J components ")
        return dic

    def enlarge_matrix(self,M,J):
        """ Enlarge a matrix by taking the Kronecker product"""
        self.__logger.info('enlarge_matrix({0},{1}'.format(M,J))

        self.__logger.debug("# Return enlarged matrices ")
        return np.kron(np.identity(M),J)
    
    # def pauli_matrix(self,M=1,op='x'):
    #     return np.kron(np.identity(M),self.angular_momentum(.5)[op])

    # def fermi_golden_rule(self):
