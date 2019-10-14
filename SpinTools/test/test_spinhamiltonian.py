import unittest
import numpy as np
from SpinTools.spinhamiltonian import spinhamiltonian as sh

class TestConstructor(unittest.TestCase):
    def test_types(self):
        # Check if raises type errors
        self.assertRaises(TypeError, sh.SpinHamiltonian,1)
        self.assertRaises(TypeError, sh.SpinHamiltonian,1j)
        self.assertRaises(TypeError, sh.SpinHamiltonian, True)
        self.assertRaises(TypeError, sh.SpinHamiltonian, None) 

class TestElectronZeeman(unittest.TestCase):
    def test_operator(self):
        ham = sh.SpinHamiltonian("Bi")
        H = ham.electron_zeeman([0,0,1])

        # Hamiltonian should have 0 trace and be self-adjoint
        self.assertTrue(np.trace(H) <= 1e-40)
        self.assertTrue((H - H.conjugate()).all() == 0)

    def test_types(self):
        # Check if raises type errors
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").electron_zeeman,"s")
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").electron_zeeman,True)
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").electron_zeeman,None)

class TestNuclearZeeman(unittest.TestCase):
    def test_operator(self):
        ham = sh.SpinHamiltonian("Bi")
        H = ham.nuclear_zeeman([0,0,1])

        # Hamiltonian should have 0 trace and be self-adjoint
        self.assertTrue(np.trace(H) <= 1e-40)
        self.assertTrue((H - H.conjugate()).all() == 0)

    def test_types(self):
        # Check if raises type errors
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").nuclear_zeeman,"s")
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").nuclear_zeeman,True)
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").nuclear_zeeman,None)

class TestHyperfine(unittest.TestCase):
    def test_operator(self):
        ham = sh.SpinHamiltonian("Bi")
        H = ham.hyperfine()

        # Hamiltonian should have 0 trace and be self-adjoint
        self.assertTrue(np.trace(H) <= 1e-40)
        self.assertTrue((H - H.conjugate()).all() == 0)

class TestGetHamiltonian(unittest.TestCase):
    def test_operator(self):
        ham = sh.SpinHamiltonian("Bi")
        H = ham.get_hamiltonian([0,0,1])

        # Hamiltonian should have 0 trace and be self-adjoint
        self.assertTrue(np.trace(H) <= 1e-40)
        self.assertTrue((H - H.conjugate()).all() == 0)

    def test_types(self):
        # Check if raises type errors
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").get_hamiltonian,"s")
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").get_hamiltonian,True)
        self.assertRaises(TypeError,sh.SpinHamiltonian("Bi").get_hamiltonian,None)
