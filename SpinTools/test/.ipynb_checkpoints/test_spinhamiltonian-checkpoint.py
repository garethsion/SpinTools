import unittest
import numpy as np
from SpinTools.spinhamiltonian import spinhamiltonian as sh

class TestConstructor(unittest.TestCase):
    def test_types(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, sh.SpinHamiltonian,1)
        self.assertRaises(TypeError, sh.SpinHamiltonian,1j)
        self.assertRaises(TypeError, sh.SpinHamiltonian, True)
        self.assertRaises(TypeError, sh.SpinHamiltonian, None) 

class TestElectronZeeman(unittest.TestCase):
	def test_hermitivity(self):
        B = [0,0,1]
        ham = sh.SpinHamiltonian("Bi")
        H = ham.electron_zeeman(B)
        Hp = H.conjugate_transpose()
        self.assertAlmostEqual(H,Hp)
        
#    def test_types(self):
#		self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").electron_zeeman,"s")
#		self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").electron_zeeman,True)
#		self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").electron_zeeman,None)
#		return